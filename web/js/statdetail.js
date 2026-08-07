/*
 * Clicking a statistic opens this: what the number means, and how it moved
 * across the whole run.
 *
 * The history comes from /api/runs/<id>/series, which reduces every frame to a
 * handful of scalars on the server. Pulling thousands of full frames into the
 * browser just to plot one line would be far slower.
 *
 * The plotted points respect the active phase filter, so looking at the game
 * phases alone gives a curve of game phases alone rather than a sawtooth
 * alternating between two different kinds of moment.
 */
const StatDetail = {
  seriesCache: new Map(),   // runId -> series payload
  currentKey: null,

  EXPLANATIONS: {
    nodes: 'How many agents are alive right now. Agents appear by reproduction and vanish by starving or being cut off from the largest component.',
    edges: 'How many connections exist. Edges are created when a newborn is wired to its parent\'s neighbourhood, and pruned when they carry no tokens through a game phase.',
    tokens: 'Total tokens in the world. This is conserved, so a flat line here is the expected behaviour — any drift would mean a bug.',
    meanDegree: 'Average number of connections per agent. Twice the edge count divided by the node count.',
    maxDegree: 'Connections held by the single most connected agent — the size of the largest hub.',
    medianTokens: 'The middle agent\'s wealth. Compared against the mean, it shows how lopsided the distribution is.',
    maxTokens: 'Wealth of the richest agent.',
    gini: 'Concentration of wealth, from 0 to 1. Zero means every agent holds exactly the same number of tokens; one means a single agent holds everything. Rising values mean wealth is pooling into fewer hands.',
    distinctBrains: 'How many different genomes are present. Every copy and every mutation mints a new brain id, so this counts distinct lineage tips rather than distinct behaviours.',
    brainDiversity: 'Distinct brains divided by node count. One means no two agents share a genome; low values mean one lineage has taken over the population.',
    births: 'Agents that reproduced this phase, each spending part of its own tokens to do so.',
    revolutions: 'Nodes taken by a coalition rather than by the single largest allocator. A revolution succeeds when the accumulated weaker allocators outweigh everyone above them plus the hegemon.',
    starved: 'Agents removed for holding zero tokens.',
    orphaned: 'Agents removed for being outside the largest connected component. They may have been perfectly wealthy — they were simply cut off.'
  },

  // Statistics that count nodes, and so are also meaningful as a percentage
  // of the population that entered the phase.
  SHARE_KEYS: new Set(['births', 'revolutions', 'starved', 'orphaned']),

  init() {
    this.el = document.getElementById('statDetail');
    this.titleEl = document.getElementById('statDetailTitle');
    this.textEl = document.getElementById('statDetailText');
    this.footEl = document.getElementById('statDetailFoot');
    this.canvas = document.getElementById('statDetailChart');

    document.getElementById('statDetailClose').addEventListener('click', () => this.close());
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape' && !this.el.classList.contains('hidden')) this.close();
    });
    window.addEventListener('resize', () => {
      if (!this.el.classList.contains('hidden')) this.redraw();
    });
  },

  close() {
    this.el.classList.add('hidden');
    this.currentKey = null;
  },

  /** Drop a run's cached series, e.g. after it has advanced. */
  invalidate(runId) {
    this.seriesCache.delete(runId);
  },

  async open(key, label) {
    if (!Viewer.runId) return;

    this.currentKey = key;
    this.titleEl.textContent = label || key;
    this.textEl.textContent = this.EXPLANATIONS[key] || 'No description for this value.';
    this.footEl.textContent = 'Loading history…';
    this.el.classList.remove('hidden');

    try {
      this.series = await this.load(Viewer.runId);
    } catch (err) {
      this.footEl.textContent = `Could not load history: ${err.message}`;
      return;
    }
    this.redraw();
  },

  async load(runId) {
    if (this.seriesCache.has(runId)) return this.seriesCache.get(runId);
    const payload = await API.getSeries(runId);
    this.seriesCache.set(runId, payload);
    return payload;
  },

  /**
   * Points for the current statistic under the active phase filter.
   *
   * Node-count statistics are converted to a share of the population that
   * entered the phase, because "40 births" means something quite different in
   * a world of 100 agents than in one of 4,000.
   */
  points() {
    const key = this.currentKey;
    if (!this.series || !key) return { xs: [], ys: [], asShare: false };

    const s = this.series.series;
    const values = s[key] || [];
    const phases = s.phase || [];
    const iterations = s.iteration || [];
    const before = s.nodes_before || [];
    const nodes = s.nodes || [];

    const asShare = this.SHARE_KEYS.has(key);
    const xs = [], ys = [];

    for (let i = 0; i < values.length; i++) {
      if (!Viewer.framePassesFilter(phases[i])) continue;
      const v = values[i];
      if (v === null || v === undefined) continue;

      if (asShare) {
        // Older runs predate nodes_before; the post-phase count is the closest
        // honest stand-in, so the curve stays usable rather than empty.
        const denominator = before[i] || nodes[i] || 0;
        if (!denominator) continue;
        ys.push((v / denominator) * 100);
      } else {
        ys.push(v);
      }
      xs.push(iterations[i]);
    }
    return { xs, ys, asShare };
  },

  redraw() {
    const { xs, ys, asShare } = this.points();
    const canvas = this.canvas;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();

    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const w = rect.width, h = rect.height;
    ctx.clearRect(0, 0, w, h);

    if (!ys.length) {
      ctx.fillStyle = '#5b6b7c';
      ctx.font = '12px system-ui, sans-serif';
      ctx.fillText('No data for this statistic under the current phase filter.', 10, h / 2);
      this.footEl.textContent = '';
      return;
    }

    const padL = 52, padR = 10, padT = 10, padB = 22;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi - lo < 1e-9) { lo -= 0.5; hi += 0.5; }

    const xAt = i => padL + (xs.length === 1 ? plotW / 2 : (i / (xs.length - 1)) * plotW);
    const yAt = v => padT + plotH - ((v - lo) / (hi - lo)) * plotH;

    // Axes
    ctx.strokeStyle = '#26313e';
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padL, padT); ctx.lineTo(padL, padT + plotH); ctx.lineTo(padL + plotW, padT + plotH);
    ctx.stroke();

    // Where the currently shown frame sits, so the number in the strip has a
    // visible home on the curve.
    const currentIteration = Viewer.frame ? Viewer.frame.iteration : null;
    if (currentIteration !== null) {
      const idx = xs.indexOf(currentIteration);
      if (idx >= 0) {
        ctx.strokeStyle = '#4fb3ff';
        ctx.globalAlpha = 0.45;
        ctx.beginPath();
        ctx.moveTo(xAt(idx), padT);
        ctx.lineTo(xAt(idx), padT + plotH);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // The curve
    ctx.strokeStyle = '#4fb3ff';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < ys.length; i++) {
      const x = xAt(i), y = yAt(ys[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    // Labels
    const fmt = v => asShare
      ? `${v.toFixed(1)}%`
      : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US') : (+v.toFixed(3)).toString());

    ctx.fillStyle = '#8fa3b5';
    ctx.font = '10px system-ui, sans-serif';
    ctx.fillText(fmt(hi), 4, padT + 8);
    ctx.fillText(fmt(lo), 4, padT + plotH);
    ctx.fillText(`iter ${xs[0]}`, padL, h - 6);
    const lastLabel = `iter ${xs[xs.length - 1]}`;
    ctx.fillText(lastLabel, padL + plotW - ctx.measureText(lastLabel).width, h - 6);

    this.footEl.textContent =
      `${ys.length} recorded ${ys.length === 1 ? 'frame' : 'frames'} · ${Viewer.phaseFilterLabel()}` +
      (asShare ? ' · shown as a share of the nodes that entered the phase' : '');
  }
};
