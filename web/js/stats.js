/*
 * Turning a frame into numbers: what drives colour, what drives size, and the
 * summary statistics and histograms shown under the canvas.
 *
 * FrameMetrics is built once per frame, and restyle() follows a change of
 * colouring, so the renderer can stay a dumb value-to-pixel mapper.
 */
class FrameMetrics {
  /**
   * `settings` are the renderer's: what colours and sizes the nodes and the
   * edges. Without them this holds values only, which is all a chart or a
   * summary reads. Diagrams used to lend it the Viewer's settings to have
   * something to pass, and so worked out, for every frame it pooled, whatever
   * the Viewer happened to be colouring by.
   */
  constructor(frame, settings = null) {
    this.frame = frame;
    this.settings = settings;

    this.index = new Map();
    frame.ids.forEach((id, i) => this.index.set(id, i));

    this.degree = this._degrees();
    // Runs recorded before deltas existed have none. The change is worked out
    // from the frame the phase started from when that is on hand, as the
    // history does; otherwise it reads as no change rather than letting
    // undefined leak into the colour maths.
    const delta = frame.delta || FrameMetrics._changeSince(frame, FrameMetrics.startOf(frame));
    this.delta = delta || new Array(frame.ids.length).fill(0);
    this.hasDelta = Boolean(delta);
    this.totalTokens = frame.tokens.reduce((a, b) => a + b, 0);
    this.curvature = this._curvature();
    if (settings) this.restyle();
  }

  /**
   * What colours and sizes the nodes, from the settings as they are now.
   *
   * Everything else here depends on the frame alone, so a change of colouring
   * recomputes these and keeps the rest: the structure, the loop counts,
   * every metric already worked out. The Viewer used to build the whole object
   * again for any such change, a colour map included, which on a large world
   * coloured by loops was two seconds a click.
   */
  restyle() {
    const settings = this.settings;
    this.colorValues = this.scaledNodeValues(settings.nodeColorBy, settings.nodeColorLog);
    this.colorRange = this.nodeRange(settings.nodeColorBy, settings.nodeColorLog);
    this.sizeValues = this.scaledNodeValues(settings.nodeSizeBy, settings.nodeSizeLog);
    this.sizeRange = this.nodeRange(settings.nodeSizeBy, settings.nodeSizeLog);

    this.colorLabel = Metrics.label('node', settings.nodeColorBy)
      + (settings.nodeColorLog ? ' (log)' : '');
    // A metric with nothing behind it in this frame — a run recorded before
    // the field existed, or a "before the phase" metric on the very first
    // frame — is every node NaN. The range then falls back to [0, 1] and every
    // node paints at mid-scale, which on screen is indistinguishable from real
    // values that happen to sit in the middle. The key says so instead, in the
    // same voice the hover card already uses for a missing token change.
    this.colorRangeText = this.hasValues(settings.nodeColorBy)
      ? [Metrics.format('node', settings.nodeColorBy, this.colorRange[0], settings.nodeColorLog),
         Metrics.format('node', settings.nodeColorBy, this.colorRange[1], settings.nodeColorLog)]
      : ['not recorded', ''];
  }

  /**
   * The frame this one's phase started from, when `frame.previous` is it, or
   * null. That is the phase just before: (it, 1) for (it, 2), and (it − 1, 2)
   * for (it, 1). gol_series.starts is the same test.
   *
   * A frame is handed the one read before it, and on a run recording every
   * Nth iteration the frame before a reproduction frame is N iterations back:
   * comparing with it would be wrong rather than merely approximate. Each
   * reader used to settle that for itself from the run's configuration, and
   * Diagrams never did.
   */
  static startOf(frame) {
    const p = frame.previous;
    if (!p || frame.iteration == null) return null;
    if (frame.phase === 2) return p.iteration === frame.iteration && p.phase === 1 ? p : null;
    if (frame.phase === 1) return p.iteration === frame.iteration - 1 && p.phase === 2 ? p : null;
    return null;
  }

  /**
   * Each node's change over the phase, from the balances it started with. A
   * node the start does not know was born during it, and counts its whole
   * balance as gained, as the engine counts a newborn's.
   */
  static _changeSince(frame, start) {
    if (!start) return null;
    const before = new Map();
    start.ids.forEach((id, i) => before.set(id, start.tokens[i]));
    return frame.ids.map((id, i) => frame.tokens[i] - (before.get(id) ?? 0));
  }

  /** Whether a node metric has any value at all in this frame. */
  hasValues(key) {
    return this.nodeValues(key).some(v => !Number.isNaN(v));
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
    // Only between agents still present, which is what the renderer draws;
    // the statistics use flowAmounts instead.
    const decisions = this.frame.decisions;
    if (!decisions || !decisions.allocations) return new Map();
    return GraphStats.flowByPair(this.frame.ids, decisions.allocations, this.index);
  }

  get flow() {
    if (!this._flow) this._flow = this._edgeFlow();
    return this._flow;
  }

  /**
   * Every amount that moved between two different agents this phase.
   *
   * Separate from `flow` on purpose. That map is keyed by position in this
   * frame because the renderer looks edges up by position, which means it can
   * only hold traffic between agents that are still here. An agent culled
   * during cleanup was still alive when the tokens were sent, so leaving its
   * traffic out understates what moved — and did, by three hundred tokens on
   * a 462-node frame, which is where this and the server's own figure parted
   * company.
   */
  get flowAmounts() {
    if (this._flowAmounts) return this._flowAmounts;

    const totals = new Map();
    const decisions = this.frame.decisions;
    if (decisions && decisions.allocations) {
      for (const record of decisions.allocations) {
        const source = record.agent;
        for (let i = 0; i < record.targets.length; i++) {
          const target = record.targets[i];
          const amount = record.alloc[i];
          if (!amount || target === source) continue;
          const key = source < target ? `${source},${target}` : `${target},${source}`;
          totals.set(key, (totals.get(key) || 0) + amount);
        }
      }
    }
    this._flowAmounts = Array.from(totals.values());
    return this._flowAmounts;
  }

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
        // A run recorded before ages were tracked has no array at all; one
        // resumed from a checkpoint that predates them has an array of -1.
        // Both are the same thing — nobody knows — so both read NaN, which
        // the charts drop and the renderer puts at mid-scale. A plausible
        // wrong number would be worse, because it cannot be questioned.
        case 'age':              out[i] = f.ages?.[i] >= 0 ? f.ages[i] : NaN; break;
        case 'node_id':          out[i] = f.ids[i]; break;
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
    return log ? this._logged('node', key, raw) : raw;
  }

  /**
   * A metric's values on a log scale, worked out once per frame. The renderer
   * asks for the edge ones on every draw, three megabytes of fresh array a
   * frame at 190,000 edges, and the node ones were computed twice each time
   * the colouring was set: once for the values, again for their range.
   */
  _logged(domain, key, raw) {
    if (!this._loggedCache) this._loggedCache = new Map();
    const cacheKey = `${domain}|${key}`;
    let out = this._loggedCache.get(cacheKey);
    if (!out) {
      const signed = Metrics.isSigned(domain, key);
      out = new Float64Array(raw.length);
      for (let i = 0; i < raw.length; i++) out[i] = Metrics.applyLog(raw[i], signed);
      this._loggedCache.set(cacheKey, out);
    }
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
    const previous = FrameMetrics.startOf(f);

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

  /**
   * What the edge colour map is showing, for the legend.
   *
   * Getters rather than fields: reading the range means walking every edge,
   * and most views never show this legend at all.
   */
  get edgeColorLabel() {
    const s = this.settings;
    return Metrics.label('edge', s.edgeColorBy) + (s.edgeColorLog ? ' (log)' : '');
  }

  get edgeColorRangeText() {
    const s = this.settings;
    const range = this.edgeRange(s.edgeColorBy, s.edgeColorLog);
    return [
      Metrics.format('edge', s.edgeColorBy, range[0], s.edgeColorLog),
      Metrics.format('edge', s.edgeColorBy, range[1], s.edgeColorLog)
    ];
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
    return log ? this._logged('edge', key, raw) : raw;
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

    this._structure = { adj, loops, triangles, dimension, distances };
    return this._structure;
  }

  /**
   * Where an edge sits in the frame's list, by its two ends, since the
   * renderer walks edges by id. The map is built the first time: only the
   * per-edge loop, triangle and bridge metrics look edges up this way, and
   * the structure used to build it on every frame whether one did or not.
   */
  edgeSlot(a, b) {
    if (!this._edgeSlots) {
      this._edgeSlots = new Map();
      this.frame.edges.forEach(([x, y], i) => this._edgeSlots.set(GraphStats.pairKey(x, y), i));
    }
    const i = this._edgeSlots.get(GraphStats.pairKey(a, b));
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
    for (const a of d.allocations || []) allocations.set(a.agent, a);
    for (const w of d.winners || []) {
      winners.set(w.node, w);
      conquests.set(w.winner, (conquests.get(w.winner) || 0) + 1);
    }

    this._decisions = { births, newborns, allocations, winners, conquests };
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

  /**
   * The numbers under the canvas.
   *
   * `includeStructure` gates the handful that need the whole graph walked;
   * without it they stay null and the strip simply omits them.
   */
  // `includeFlow` defaults to whatever `includeStructure` is, so a caller that
  // wants everything still says so with one argument.
  summary(includeStructure = true, includeFlow = includeStructure) {
    const f = this.frame;
    const n = f.ids.length;
    const degrees = Array.from(this.degree);
    const tokens = f.tokens;
    const d = f.decisions || {};

    const sum = arr => arr.reduce((a, b) => a + b, 0);
    const mean = arr => arr.length ? sum(arr) / arr.length : 0;
    const middle = s => {
      if (!s.length) return 0;
      const m = Math.floor(s.length / 2);
      return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
    };
    const median = arr => middle([...arr].sort((a, b) => a - b));
    // One pass rather than Math.max(...arr): a spread hands every value over
    // as an argument, and past about 125,000 of them the engine throws.
    const largest = arr => { let m = -Infinity; for (const v of arr) if (v > m) m = v; return m; };
    const smallest = arr => { let m = Infinity; for (const v of arr) if (v < m) m = v; return m; };

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
    const distinctParents = new Set(f.parent_brain_ids).size;

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
      maxDegree: degrees.length ? largest(degrees) : 0,
      minDegree: degrees.length ? smallest(degrees) : 0,
      leaves: degrees.filter(x => x === 1).length,

      // Wealth
      tokens: this.totalTokens,
      meanTokens: mean(Array.from(tokens)),
      // The sorted copy the Gini coefficient needed anyway, not a second sort.
      medianTokens: middle(sorted),
      maxTokens: tokens.length ? largest(tokens) : 0,
      minTokens: tokens.length ? smallest(tokens) : 0,
      gini,
      topDecileShare: topShare,

      // Biggest single swing either way this phase. Losses are reported as a
      // positive magnitude so the two read side by side.
      maxTokenAdded: this.delta.length ? Math.max(0, largest(this.delta)) : 0,
      maxTokenLost: this.delta.length ? Math.max(0, -smallest(this.delta)) : 0,
      gainers: this.delta.filter(v => v > 0).length,
      losers: this.delta.filter(v => v < 0).length,

      // Structure
      degreeExponent: null, degreeExponentR2: null,
      tokenExponent: null, tokenExponentR2: null,
      tokensVsDegree: null, tokensVsDegreeR2: null,
      trianglesVsDegree: null, trianglesVsDegreeR2: null,
      clusteringVsDegree: null, clusteringVsDegreeR2: null,
      changeVsTokens: null, changeVsTokensR2: null,
      assortativity: null,
      degreeGamma: null, degreeGammaR2: null, degreeKMin: null,
      degreeTailShare: null, degreeGammaKS: null,
      boxDimension: null, boxDimensionR2: null,
      cycleRank: null, loopDensity: null, bridges: null, triangles: null,
      cutRisk: null, coreShare: null, spectralGap: null, ricciCurvature: null,
      lightningScore: null, cyclingShare: null,
      lightningLongest: null, flowImbalance: null,
      netLightningScore: null, netCyclingShare: null,
      netLightningLongest: null, netFlowShare: null,
      transitivity: null, degreeEntropy: null, degreeEvenness: null,
      radius: null, diameter: null, meanPathLength: null,
      tokenEntropy: null, tokenEvenness: null, dimension: null, components: null,

      // Genome
      distinctBrains,
      brainDiversity: n ? distinctBrains / n : 0,
      distinctParents,

      // Cleanup, present on both phases
      starved: f.cleanup ? f.cleanup.starved : null,
      orphaned: f.cleanup ? f.cleanup.orphaned : null,
      redistributed: f.cleanup ? f.cleanup.redistributed : null,
      // Taken by the engine before the cull, so it can be read as a cause of
      // the cull rather than a consequence. Runs made before the engine
      // recorded it simply do not have it.
      cutRiskBefore: f.cleanup && f.cleanup.cutRiskBefore !== undefined
        ? f.cleanup.cutRiskBefore : null,

      births: null, meanInvestedShare: null, meanChildLinks: null,
      reproTokenShare: null, handovers: null,
      gifts: null, giftTokens: null, giftShare: null,
      revolutions: null, totalFlow: null, meanEdgeFlow: null, maxEdgeFlow: null,
      selfAllocationShare: null, revoltShare: null, spreadShare: null,
      heldHomeShare: null, prunedEdges: null
    };

    // Rewiring happens during the reproduction phase but is nobody's child:
    // absent when the rule is off, rather than a zero that reads as "nobody
    // chose to".

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

    // Gifts. Absent rather than zero on a run without the mechanic, so a world
    // where nobody chose to give reads differently from one where nobody could.
    if (d.gifts) {
      out.gifts = d.gifts.length;
      out.giftTokens = sum(d.gifts.map(g => g[2]));
      out.giftShare = this.totalTokens ? out.giftTokens / this.totalTokens : 0;
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

      const flows = this.flowAmounts;
      out.totalFlow = sum(flows);
      out.meanEdgeFlow = mean(flows);
      out.maxEdgeFlow = flows.length ? largest(flows) : 0;
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

    // ---- circulating token flow ----
    //
    // Gated separately from the structure block below. The lightning search
    // reads this phase's allocations and nothing else — it never touches the
    // graph walk — so tying it to the structure group would have made it wait
    // on a walk it does not use, and would leave it blank on a strip where the
    // group holding it is open. Only a game phase allocates across links, so a
    // reproduction frame has no lightning and says so with nulls.
    // Kept with the frame's other results: the strip asks again every time a
    // group is opened or closed, and it is the same frame each time.
    if (includeFlow) Object.assign(out, (this._lightning ||= Lightning.of(f)));

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
      out.cutRisk = st.loops.cutRisk;
      out.coreShare = st.loops.coreShare;
      out.spectralGap = st.loops.spectralGap;
      out.components = st.loops.componentCount;
      out.triangles = st.triangles.total;
      out.transitivity = GraphStats.transitivity(f.ids, st.adj, st.triangles.total);
      out.dimension = st.dimension.estimate;
      out.ricciCurvature = st.dimension.ricciCurvature;

      out.radius = st.distances.radius;
      out.diameter = st.distances.diameter;
      out.meanPathLength = st.distances.meanPathLength;

      // ---- power laws ----
      //
      // How one quantity scales with another, as an exponent and how tightly
      // the points sit on that line. Mirrors the same block in gol_series.py,
      // which tests/test_stats_parity.py compares value for value.
      const ids = f.ids;
      const degrees = [], tokensList = [], triangleList = [];
      for (let i = 0; i < ids.length; i++) {
        degrees.push(this.degree[i]);
        tokensList.push(f.tokens[i]);
        triangleList.push(st.triangles.perNode.get(ids[i]) || 0);
      }

      const clustering = GraphStats.clusteringPerNode(ids, st.adj, st.triangles.perNode);
      const clusteringDegrees = [], clusteringValues = [];
      for (const id of ids) {
        if (!clustering.has(id)) continue;   // fewer than two neighbours
        clusteringDegrees.push((st.adj.get(id) || []).size || 0);
        clusteringValues.push(clustering.get(id));
      }

      const changeTokens = [], changeSizes = [];
      for (let i = 0; i < Math.min(tokensList.length, this.delta.length); i++) {
        changeTokens.push(tokensList[i]);
        changeSizes.push(Math.abs(this.delta[i]));
      }

      const pair = (fit, key) => {
        out[key] = fit ? fit.exponent : null;
        out[key + 'R2'] = fit ? fit.r2 : null;
      };
      pair(GraphStats.tailExponent(degrees), 'degreeExponent');
      pair(GraphStats.tailExponent(tokensList), 'tokenExponent');
      pair(GraphStats.powerFit(degrees, tokensList), 'tokensVsDegree');
      pair(GraphStats.powerFit(degrees, triangleList), 'trianglesVsDegree');
      pair(GraphStats.powerFit(clusteringDegrees, clusteringValues), 'clusteringVsDegree');
      pair(GraphStats.powerFit(changeTokens, changeSizes), 'changeVsTokens');

      const degreeOf = new Map();
      for (let i = 0; i < ids.length; i++) degreeOf.set(ids[i], this.degree[i]);
      out.assortativity = GraphStats.assortativity(f.edges, id => degreeOf.get(id));

      // Scale free: the degree distribution's tail, found rather than assumed.
      const sf = GraphStats.scaleFree(degrees);
      out.degreeGamma = sf ? sf.exponent : null;
      out.degreeGammaR2 = sf ? sf.r2 : null;
      out.degreeKMin = sf ? sf.kMin : null;
      out.degreeTailShare = sf ? sf.coverage : null;
      out.degreeGammaKS = sf ? sf.ks : null;

      // Self-similar: how the number of boxes needed falls as boxes grow.
      const boxes = GraphStats.boxDimension(ids, st.adj);
      out.boxDimension = boxes ? boxes.exponent : null;
      out.boxDimensionR2 = boxes ? boxes.r2 : null;
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
