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
    this.flow = this._edgeFlow();
    this.totalTokens = frame.tokens.reduce((a, b) => a + b, 0);

    this.colorValues = this._values(settings.nodeColorBy);
    this.colorRange = this._range(this.colorValues);
    this.sizeValues = this._values(settings.nodeSizeBy);
    this.sizeRange = this._range(this.sizeValues);

    this.colorLabel = FrameMetrics.LABELS[settings.nodeColorBy] || settings.nodeColorBy;
    this.colorRangeText = [
      FrameMetrics.formatValue(this.colorRange[0], settings.nodeColorBy),
      FrameMetrics.formatValue(this.colorRange[1], settings.nodeColorBy)
    ];
  }

  static LABELS = {
    tokens: 'Tokens',
    log_tokens: 'Tokens (log)',
    degree: 'Degree',
    log_degree: 'Degree (log)',
    brain_id: 'Brain id',
    parent_brain_id: 'Parent brain id',
    age: 'Node id (age)',
    token_share: 'Share of tokens',
    constant: 'Constant'
  };

  static formatValue(v, kind) {
    if (kind === 'log_tokens' || kind === 'log_degree') {
      return Math.round(Math.expm1(v)).toLocaleString('en-US');
    }
    if (kind === 'token_share') return `${(v * 100).toFixed(2)}%`;
    return Math.round(v).toLocaleString('en-US');
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

    for (const record of decisions.allocations) {
      const source = record.agent;
      for (let i = 0; i < record.targets.length; i++) {
        const target = record.targets[i];
        const amount = record.alloc[i];
        if (!amount || target === source) continue;
        const key = source < target ? `${source},${target}` : `${target},${source}`;
        flow.set(key, (flow.get(key) || 0) + amount);
      }
    }
    return flow;
  }

  get hasFlow() { return this.flow.size > 0; }

  _values(kind) {
    const f = this.frame;
    const n = f.ids.length;
    const out = new Float64Array(n);

    for (let i = 0; i < n; i++) {
      switch (kind) {
        case 'tokens':          out[i] = f.tokens[i]; break;
        case 'log_tokens':      out[i] = Math.log1p(f.tokens[i]); break;
        case 'degree':          out[i] = this.degree[i]; break;
        case 'log_degree':      out[i] = Math.log1p(this.degree[i]); break;
        case 'brain_id':        out[i] = f.brain_ids[i]; break;
        case 'parent_brain_id': out[i] = f.parent_brain_ids[i]; break;
        case 'age':             out[i] = f.ids[i]; break;
        case 'token_share':     out[i] = this.totalTokens ? f.tokens[i] / this.totalTokens : 0; break;
        default:                out[i] = 0.5;
      }
    }
    return out;
  }

  _range(values) {
    let lo = Infinity, hi = -Infinity;
    for (const v of values) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!Number.isFinite(lo)) return [0, 1];
    return (hi - lo < 1e-9) ? [lo, lo + 1] : [lo, hi];
  }

  static _norm(v, [lo, hi]) {
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

    // Log variants share the underlying quantity; only the scale differs, and
    // that is applied once in _edgeNorm rather than duplicated per case.
    switch (FrameMetrics.baseEdgeKind(kind)) {
      case 'avg_tokens': return (f.tokens[ia] + f.tokens[ib]) / 2;
      case 'min_tokens': return Math.min(f.tokens[ia], f.tokens[ib]);
      case 'max_tokens': return Math.max(f.tokens[ia], f.tokens[ib]);
      case 'avg_degree': return (this.degree[ia] + this.degree[ib]) / 2;
      case 'flow': {
        const key = a < b ? `${a},${b}` : `${b},${a}`;
        return this.flow.get(key) || 0;
      }
      default: return 0;
    }
  }

  /** Strip a leading `log_` to get the quantity underneath. */
  static baseEdgeKind(kind) {
    return kind.startsWith('log_') ? kind.slice(4) : kind;
  }

  _edgeRange(kind) {
    if (!this._edgeRanges) this._edgeRanges = new Map();
    if (this._edgeRanges.has(kind)) return this._edgeRanges.get(kind);

    let lo = Infinity, hi = -Infinity;
    for (const [a, b] of this.frame.edges) {
      const v = this._edgeRaw(kind, a, b);
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    const range = (!Number.isFinite(lo) || hi - lo < 1e-9) ? [0, 1] : [lo, hi];
    this._edgeRanges.set(kind, range);
    return range;
  }

  /**
   * Normalised value for an edge under the given mode.
   *
   * Raw flow is heavily skewed — a handful of edges carry most of the tokens —
   * so it is always read on a log scale; everything else offers log as an
   * explicit `log_` option.
   */
  _edgeNorm(kind, a, b) {
    if (kind === 'constant' || kind === 'source') return 0.5;

    const range = this._edgeRange(kind);
    const value = this._edgeRaw(kind, a, b);

    if (kind === 'flow' || kind.startsWith('log_')) {
      return FrameMetrics._norm(Math.log1p(Math.max(0, value)),
                                [Math.log1p(Math.max(0, range[0])),
                                 Math.log1p(Math.max(0, range[1]))]);
    }
    return FrameMetrics._norm(value, range);
  }

  edgeColorNorm(a, b) {
    return this._edgeNorm(this.settings.edgeColorBy, a, b);
  }

  edgeWidthNorm(a, b) {
    const kind = this.settings.edgeWidthBy;
    if (kind === 'constant') return 0;
    return this._edgeNorm(kind, a, b);
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
      phase: f.phase
    };

    if (f.phase === 1) {
      const birth = idx.births.get(id);
      detail.reproduced = idx.births.size ? Boolean(birth) : null;
      if (birth) {
        detail.invested = birth.invested;
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

  /** Tokens carried by each edge this phase, as a plain array. */
  flowValues() {
    return Array.from(this.flow.values());
  }

  summary() {
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

      // Genome
      distinctBrains,
      brainDiversity: n ? distinctBrains / n : 0,
      distinctLineages,

      // Cleanup, present on both phases
      starved: f.cleanup ? f.cleanup.starved : null,
      orphaned: f.cleanup ? f.cleanup.orphaned : null,
      redistributed: f.cleanup ? f.cleanup.redistributed : null,

      births: null, meanInvestedShare: null, meanChildLinks: null,
      revolutions: null, totalFlow: null, meanEdgeFlow: null, maxEdgeFlow: null,
      selfAllocationShare: null, revoltShare: null, spreadShare: null,
      heldHomeShare: null, prunedEdges: null
    };

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
      out.revoltShare = allocatedTotal ? revolted / allocatedTotal : 0;
      out.spreadShare = allocations.length ? spreadCount / allocations.length : 0;

      const flows = this.flowValues();
      out.totalFlow = sum(flows);
      out.meanEdgeFlow = mean(flows);
      out.maxEdgeFlow = flows.length ? Math.max(...flows) : 0;
    }

    if (d.winners) {
      out.revolutions = d.winners.filter(w => w.revolt).length;
      out.heldHomeShare = d.winners.length
        ? d.winners.filter(w => w.winner === w.node).length / d.winners.length
        : 0;
    }

    if (d.pruned_edges) out.prunedEdges = d.pruned_edges.length;

    return out;
  }
}

// --------------------------------------------------------------------------
// Histograms
// --------------------------------------------------------------------------

/**
 * Draw a bar histogram into a canvas.
 * `logScale` bins on log1p, which is what makes the token distribution legible
 * when a handful of agents hold most of the economy.
 */
function drawHistogram(canvas, values, options = {}) {
  const { bins = 32, colormap = 'viridis', reverse = false, logScale = false } = options;
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const w = rect.width, h = rect.height;
  ctx.clearRect(0, 0, w, h);

  if (!values || !values.length) {
    ctx.fillStyle = '#5b6b7c';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText('no data', 8, h / 2);
    return;
  }

  const mapped = logScale ? Array.from(values, v => Math.log1p(Math.max(0, v))) : Array.from(values);
  let lo = Math.min(...mapped), hi = Math.max(...mapped);
  if (hi - lo < 1e-9) hi = lo + 1;

  const counts = new Array(bins).fill(0);
  for (const v of mapped) {
    const b = Math.min(bins - 1, Math.floor((v - lo) / (hi - lo) * bins));
    counts[b]++;
  }
  const peak = Math.max(...counts) || 1;

  const padBottom = 16, padTop = 6;
  const plotH = h - padBottom - padTop;
  const barW = w / bins;

  for (let i = 0; i < bins; i++) {
    const barH = (counts[i] / peak) * plotH;
    ctx.fillStyle = colormapCss(colormap, i / (bins - 1), 0.92, reverse);
    ctx.fillRect(i * barW, padTop + plotH - barH, Math.max(1, barW - 1), barH);
  }

  // Axis: lowest and highest bin edge, in original units.
  ctx.fillStyle = '#8fa3b5';
  ctx.font = '10px system-ui, sans-serif';
  const back = v => logScale ? Math.expm1(v) : v;
  const fmt = v => Math.round(back(v)).toLocaleString('en-US');
  ctx.fillText(fmt(lo), 2, h - 4);
  const hiText = fmt(hi);
  ctx.fillText(hiText, w - ctx.measureText(hiText).width - 2, h - 4);
  const peakText = `peak ${peak}`;
  ctx.fillText(peakText, (w - ctx.measureText(peakText).width) / 2, h - 4);
}
