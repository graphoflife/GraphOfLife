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
    brain_id: 'Brain id',
    parent_brain_id: 'Parent brain id',
    age: 'Node id (age)',
    token_share: 'Share of tokens',
    constant: 'Constant'
  };

  static formatValue(v, kind) {
    if (kind === 'log_tokens') return Math.round(Math.expm1(v)).toLocaleString('en-US');
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

    switch (kind) {
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

  edgeColorNorm(a, b) {
    const kind = this.settings.edgeColorBy;
    if (kind === 'constant' || kind === 'source') return 0.5;
    // Flow is heavily skewed — a log scale keeps the busiest edges from
    // flattening everything else into a single colour.
    if (kind === 'flow') {
      const range = this._edgeRange(kind);
      return FrameMetrics._norm(Math.log1p(this._edgeRaw(kind, a, b)),
                                [Math.log1p(range[0]), Math.log1p(range[1])]);
    }
    return FrameMetrics._norm(this._edgeRaw(kind, a, b), this._edgeRange(kind));
  }

  edgeWidthNorm(a, b) {
    const kind = this.settings.edgeWidthBy;
    if (kind === 'constant') return 0;
    if (kind === 'flow') {
      const range = this._edgeRange(kind);
      return FrameMetrics._norm(Math.log1p(this._edgeRaw(kind, a, b)),
                                [Math.log1p(range[0]), Math.log1p(range[1])]);
    }
    return FrameMetrics._norm(this._edgeRaw(kind, a, b), this._edgeRange(kind));
  }

  // ---- summary --------------------------------------------------------

  summary() {
    const f = this.frame;
    const n = f.ids.length;
    const degrees = Array.from(this.degree);
    const tokens = f.tokens;

    const mean = arr => arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
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

    const distinctBrains = new Set(f.brain_ids).size;

    return {
      iteration: f.iteration,
      phase: f.phase,
      nodes: n,
      edges: f.edges.length,
      tokens: this.totalTokens,
      meanDegree: mean(degrees),
      maxDegree: degrees.length ? Math.max(...degrees) : 0,
      medianTokens: median(tokens),
      maxTokens: tokens.length ? Math.max(...tokens) : 0,
      gini,
      distinctBrains,
      brainDiversity: n ? distinctBrains / n : 0,
      births: (f.decisions && f.decisions.births) ? f.decisions.births.length : null,
      revolutions: (f.decisions && f.decisions.winners)
        ? f.decisions.winners.filter(w => w.revolt).length : null,
      starved: f.cleanup ? f.cleanup.starved : null,
      orphaned: f.cleanup ? f.cleanup.orphaned : null
    };
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
