/*
 * Everything under the canvas: the playbar's label, the statistics, the three
 * charts, and the hover card.
 *
 * These share a shape — take the frame on screen, reduce it to something a
 * person can read, put it in the page — and none of them feed back into the
 * simulation or the layout. Keeping them here leaves viewer.js about frames,
 * cameras and time.
 */

Object.assign(Viewer, {

  // ------------------------------------------------------------------
  // Presentation
  // ------------------------------------------------------------------

  updateSlider() {
    const slider = document.getElementById('frameSlider');
    slider.max = Math.max(0, this.visible.length - 1);
    slider.value = this.position;

    // Written into three fixed-width slots rather than one string, so the bar
    // cannot reflow as the numbers change width while playing.
    const iterEl = document.getElementById('flIter');
    const phaseEl = document.getElementById('flPhase');
    const posEl = document.getElementById('flPos');

    if (this.frame) {
      iterEl.textContent = `Iteration ${formatNumber(this.frame.iteration)}`;
      phaseEl.textContent = this.frame.phase === 1 ? 'reproduction' : 'game';
      posEl.textContent = `${this.position + 1}/${this.visible.length}`;
    } else {
      iterEl.textContent = '—';
      phaseEl.textContent = '';
      posEl.textContent = '';
    }
  },

  /**
   * Which category each statistic belongs to, and the order within it.
   *
   * Phase-specific groups simply come out empty on the other phase, so the
   * Reproduction section disappears on a game frame rather than showing a row
   * of dashes.
   */
  /**
   * What each statistic is called, in one place.
   *
   * The strip under the canvas and the trajectory chart's menus both read
   * from here, so a statistic cannot end up with two different names
   * depending on where you look at it.
   */
  STAT_LABELS: {
    nodes: 'Nodes',
    edges: 'Edges',
    tokens: 'Tokens',
    meanTokens: 'Mean tokens',
    medianTokens: 'Median tokens',
    maxTokens: 'Richest',
    minTokens: 'Poorest',
    gini: 'Gini',
    topDecileShare: 'Top 10% hold',
    tokenEntropy: 'Token entropy',
    tokenEvenness: 'Token evenness',
    maxTokenAdded: 'Max token added',
    maxTokenLost: 'Max token lost',
    gainers: 'Gained',
    losers: 'Lost',
    distinctBrains: 'Distinct brains',
    brainDiversity: 'Brain diversity',
    distinctParents: 'Parent genomes',
    density: 'Density',
    meanDegree: 'Mean degree',
    medianDegree: 'Median degree',
    maxDegree: 'Max degree',
    minDegree: 'Min degree',
    leaves: 'Leaves',
    degreeEntropy: 'Degree entropy',
    degreeEvenness: 'Degree evenness',
    cycleRank: 'Loops',
    loopDensity: 'Loop density',
    bridges: 'Bridges',
    triangles: 'Triangles',
    transitivity: 'Clustering',
    dimension: 'Dimension',
    radius: 'Radius',
    diameter: 'Diameter',
    meanPathLength: 'Mean path',
    components: 'Components',

    degreeGamma: 'Scale-free \u03b3',
    degreeGammaR2: 'Scale-free R\u00b2',
    degreeKMin: 'Tail starts at k',
    degreeTailShare: 'Tail share',
    degreeGammaKS: 'Scale-free KS',
    boxDimension: 'Box dimension d\u1d47',
    boxDimensionR2: 'Box dimension R\u00b2',

    degreeExponent: 'Degree exponent \u03b3',
    degreeExponentR2: 'Degree fit R\u00b2',
    tokenExponent: 'Token exponent \u03b3',
    tokenExponentR2: 'Token fit R\u00b2',
    tokensVsDegree: 'Tokens vs degree',
    tokensVsDegreeR2: 'Tokens vs degree R\u00b2',
    trianglesVsDegree: 'Triangles vs degree',
    trianglesVsDegreeR2: 'Triangles vs degree R\u00b2',
    clusteringVsDegree: 'Clustering vs degree',
    clusteringVsDegreeR2: 'Clustering vs degree R\u00b2',
    changeVsTokens: 'Token change vs tokens',
    changeVsTokensR2: 'Token change vs tokens R\u00b2',
    assortativity: 'Assortativity',
    births: 'Births',
    reproTokenShare: 'Tokens to offspring',
    meanInvestedShare: 'Mean investment',
    meanChildLinks: 'Links per child',
    handovers: 'Handovers',
    totalFlow: 'Tokens moved',
    meanEdgeFlow: 'Mean edge flow',
    maxEdgeFlow: 'Max edge flow',
    selfAllocationShare: 'Kept at home',
    spreadShare: 'Spread doctrine',
    revoltShare: 'Revolt tokens',
    revolutions: 'Revolutions',
    heldHomeShare: 'Held own node',
    prunedEdges: 'Pruned edges',
    starved: 'Starved',
    orphaned: 'Culled',
    redistributed: 'Redistributed'
  },

  STAT_GROUPS: [
    { key: 'general', label: 'General', open: true, keys: [
      'nodes', 'edges', 'tokens', 'meanTokens', 'medianTokens', 'maxTokens', 'minTokens',
      'gini', 'topDecileShare', 'tokenEntropy', 'tokenEvenness',
      'maxTokenAdded', 'maxTokenLost', 'gainers', 'losers',
      'starved', 'orphaned', 'redistributed',
      'distinctBrains', 'brainDiversity', 'distinctParents'
    ] },
    { key: 'reproduction', label: 'Reproduction', open: true, keys: [
      'births', 'reproTokenShare', 'meanInvestedShare', 'meanChildLinks',
      'handovers'
    ] },
    { key: 'blotto', label: 'Game (Blotto)', open: true, keys: [
      'totalFlow', 'meanEdgeFlow', 'maxEdgeFlow', 'selfAllocationShare',
      'revoltShare', 'spreadShare', 'revolutions', 'heldHomeShare', 'prunedEdges'
    ] },
    { key: 'structure', label: 'Structure', open: false, keys: [
      'density', 'meanDegree', 'medianDegree', 'maxDegree', 'minDegree', 'leaves',
      'radius', 'diameter', 'meanPathLength',
      'cycleRank', 'loopDensity', 'bridges', 'triangles', 'transitivity',
      'dimension', 'degreeEntropy', 'degreeEvenness', 'components'
    ] },
    { key: 'powerlaws', label: 'Power laws', open: false, keys: [
      'degreeGamma', 'degreeGammaR2', 'degreeKMin', 'degreeTailShare', 'degreeGammaKS',
      'boxDimension', 'boxDimensionR2',
      'degreeExponent', 'degreeExponentR2',
      'tokenExponent', 'tokenExponentR2',
      'tokensVsDegree', 'tokensVsDegreeR2',
      'trianglesVsDegree', 'trianglesVsDegreeR2',
      'clusteringVsDegree', 'clusteringVsDegreeR2',
      'changeVsTokens', 'changeVsTokensR2',
      'assortativity'
    ] }
  ],

  updateStats() {
    const container = document.getElementById('statsStrip');
    if (!this.metrics) { container.innerHTML = ''; return; }

    // Whether the reader currently has the Structure group open decides
    // whether its statistics are worth computing at all: they cost more than
    // everything else on this strip put together.
    // Structure and Power laws are both paid for by the same walk over the
    // graph, so either being open buys both. Neither being open means the walk
    // does not happen at all, which is most of what a frame step costs.
    const heavyGroups = ['structure', 'powerlaws'];
    const structureOpen = heavyGroups.some(key => {
      const el = container.querySelector(`.stat-group[data-group="${key}"]`);
      if (el) return el.open;
      const declared = this.STAT_GROUPS.find(g => g.key === key);
      return Boolean(declared && declared.open);
    });
    const s = this.metrics.summary(structureOpen);

    // Node counts are also given as a share of the population that entered the
    // phase — "40 births" reads very differently at 100 agents than at 4,000.
    const base = s.nodesBefore || s.nodes || 0;
    const withShare = v => (base && v !== null && v !== undefined)
      ? `${formatNumber(v)} <i>${((v / base) * 100).toFixed(1)}%</i>` : formatNumber(v);

    const int = v => (v === null || v === undefined) ? '\u2014' : formatNumber(Math.round(v));
    const pct = v => (v === null || v === undefined) ? '\u2014' : `${(v * 100).toFixed(1)}%`;
    const dec = (v, n = 2) => (v === null || v === undefined) ? '\u2014' : v.toFixed(n);

    // label and formatted value for every statistic that has one this frame
    const cells = {
      nodes: [this.STAT_LABELS.nodes, formatNumber(s.nodes)],
      edges: [this.STAT_LABELS.edges, formatNumber(s.edges)],
      tokens: [this.STAT_LABELS.tokens, formatNumber(s.tokens)],
      meanTokens: [this.STAT_LABELS.meanTokens, int(s.meanTokens)],
      medianTokens: [this.STAT_LABELS.medianTokens, int(s.medianTokens)],
      maxTokens: [this.STAT_LABELS.maxTokens, formatNumber(s.maxTokens)],
      minTokens: [this.STAT_LABELS.minTokens, formatNumber(s.minTokens)],
      gini: [this.STAT_LABELS.gini, dec(s.gini, 3)],
      topDecileShare: [this.STAT_LABELS.topDecileShare, pct(s.topDecileShare)],
      tokenEntropy: [this.STAT_LABELS.tokenEntropy, `${dec(s.tokenEntropy)} bits`],
      tokenEvenness: [this.STAT_LABELS.tokenEvenness, pct(s.tokenEvenness)],
      maxTokenAdded: [this.STAT_LABELS.maxTokenAdded, `+${formatNumber(s.maxTokenAdded)}`],
      maxTokenLost: [this.STAT_LABELS.maxTokenLost, `-${formatNumber(s.maxTokenLost)}`],
      gainers: [this.STAT_LABELS.gainers, withShare(s.gainers)],
      losers: [this.STAT_LABELS.losers, withShare(s.losers)],
      distinctBrains: [this.STAT_LABELS.distinctBrains, formatNumber(s.distinctBrains)],
      brainDiversity: [this.STAT_LABELS.brainDiversity, pct(s.brainDiversity)],
      distinctParents: [this.STAT_LABELS.distinctParents, formatNumber(s.distinctParents)],

      density: [this.STAT_LABELS.density, `${(s.density * 100).toFixed(2)}%`],
      meanDegree: [this.STAT_LABELS.meanDegree, dec(s.meanDegree)],
      medianDegree: [this.STAT_LABELS.medianDegree, dec(s.medianDegree, 1)],
      maxDegree: [this.STAT_LABELS.maxDegree, formatNumber(s.maxDegree)],
      minDegree: [this.STAT_LABELS.minDegree, formatNumber(s.minDegree)],
      leaves: [this.STAT_LABELS.leaves, withShare(s.leaves)],
      degreeEntropy: [this.STAT_LABELS.degreeEntropy, `${dec(s.degreeEntropy)} bits`],
      degreeEvenness: [this.STAT_LABELS.degreeEvenness, pct(s.degreeEvenness)]
    };

    // Always listed, even before they are computed. A group with no cells is
    // not rendered, and a group that is never rendered can never be opened to
    // ask for the numbers it would hold — so these show a dash until the group
    // is opened and the walk over the graph has been paid for.
    //
    // Exponents are slopes and carry a sign, so they keep it rather than being
    // rounded into a bare magnitude. R² is a share of the spread accounted
    // for, so it reads as a percentage.
    const exponent = v => (v === null || v === undefined)
      ? '\u2014' : (v > 0 ? '+' : '') + v.toFixed(2);
    for (const key of ['degreeExponent', 'tokenExponent', 'tokensVsDegree',
                       'trianglesVsDegree', 'clusteringVsDegree', 'changeVsTokens']) {
      cells[key] = [this.STAT_LABELS[key], exponent(s[key])];
      cells[key + 'R2'] = [this.STAT_LABELS[key + 'R2'], pct(s[key + 'R2'])];
    }
    cells.assortativity = [this.STAT_LABELS.assortativity, dec(s.assortativity, 3)];
    cells.degreeGamma = [this.STAT_LABELS.degreeGamma, dec(s.degreeGamma)];
    cells.degreeGammaR2 = [this.STAT_LABELS.degreeGammaR2, pct(s.degreeGammaR2)];
    cells.degreeKMin = [this.STAT_LABELS.degreeKMin, int(s.degreeKMin)];
    cells.degreeTailShare = [this.STAT_LABELS.degreeTailShare, pct(s.degreeTailShare)];
    cells.degreeGammaKS = [this.STAT_LABELS.degreeGammaKS, dec(s.degreeGammaKS, 3)];
    cells.boxDimension = [this.STAT_LABELS.boxDimension, dec(s.boxDimension)];
    cells.boxDimensionR2 = [this.STAT_LABELS.boxDimensionR2, pct(s.boxDimensionR2)];

    // Only computed while one of the heavy groups is open, since walking the
    // whole graph costs more than the rest of this strip together.
    if (structureOpen) {
      cells.cycleRank = [this.STAT_LABELS.cycleRank, formatNumber(s.cycleRank)];
      cells.loopDensity = [this.STAT_LABELS.loopDensity, pct(s.loopDensity)];
      cells.bridges = [this.STAT_LABELS.bridges, formatNumber(s.bridges)];
      cells.triangles = [this.STAT_LABELS.triangles, formatNumber(s.triangles)];
      cells.transitivity = [this.STAT_LABELS.transitivity, dec(s.transitivity, 3)];
      cells.dimension = [this.STAT_LABELS.dimension, dec(s.dimension)];
      cells.radius = [this.STAT_LABELS.radius, formatNumber(s.radius)];
      cells.diameter = [this.STAT_LABELS.diameter, formatNumber(s.diameter)];
      cells.meanPathLength = [this.STAT_LABELS.meanPathLength, dec(s.meanPathLength)];
      cells.components = [this.STAT_LABELS.components, formatNumber(s.components)];

    }

    // Present only when the phase produced them.
    if (s.births !== null) {
      cells.births = [this.STAT_LABELS.births, withShare(s.births)];
      cells.reproTokenShare = [this.STAT_LABELS.reproTokenShare, pct(s.reproTokenShare)];
      cells.meanInvestedShare = [this.STAT_LABELS.meanInvestedShare, pct(s.meanInvestedShare)];
      cells.meanChildLinks = [this.STAT_LABELS.meanChildLinks, dec(s.meanChildLinks)];
      if (s.handovers !== null) cells.handovers = [this.STAT_LABELS.handovers, formatNumber(s.handovers)];
    }
    if (s.totalFlow !== null) {
      cells.totalFlow = [this.STAT_LABELS.totalFlow, formatNumber(s.totalFlow)];
      cells.meanEdgeFlow = [this.STAT_LABELS.meanEdgeFlow, dec(s.meanEdgeFlow, 1)];
      cells.maxEdgeFlow = [this.STAT_LABELS.maxEdgeFlow, formatNumber(s.maxEdgeFlow)];
      cells.selfAllocationShare = [this.STAT_LABELS.selfAllocationShare, pct(s.selfAllocationShare)];
      cells.spreadShare = [this.STAT_LABELS.spreadShare, pct(s.spreadShare)];
      // Null when the run has revolutions off. Formatting that as 0% would
      // claim nobody revolted, when in fact nobody could.
      if (s.revoltShare !== null) cells.revoltShare = [this.STAT_LABELS.revoltShare, pct(s.revoltShare)];
    }
    if (s.revolutions !== null) cells.revolutions = [this.STAT_LABELS.revolutions, withShare(s.revolutions)];
    if (s.heldHomeShare !== null) cells.heldHomeShare = [this.STAT_LABELS.heldHomeShare, pct(s.heldHomeShare)];
    if (s.prunedEdges !== null) cells.prunedEdges = [this.STAT_LABELS.prunedEdges, formatNumber(s.prunedEdges)];
    if (s.starved !== null) cells.starved = [this.STAT_LABELS.starved, withShare(s.starved)];
    if (s.orphaned !== null) cells.orphaned = [this.STAT_LABELS.orphaned, withShare(s.orphaned)];
    if (s.redistributed !== null) cells.redistributed = [this.STAT_LABELS.redistributed, formatNumber(s.redistributed)];

    // Remember which sections were open, so redrawing a frame does not fold
    // everything back up under the reader.
    const wasOpen = new Map();
    for (const el of container.querySelectorAll('.stat-group')) {
      wasOpen.set(el.dataset.group, el.open);
    }

    const html = [];
    for (const group of this.STAT_GROUPS) {
      const present = group.keys.filter(k => cells[k]);
      if (!present.length) continue;

      const open = wasOpen.has(group.key) ? wasOpen.get(group.key) : group.open;
      const body = present.map(k => {
        const [label, value] = cells[k];
        return `<button class="stat" data-stat="${k}" data-label="${label}"
                  title="Click for an explanation and its history">
                  <span class="stat-key">${label}</span><span class="stat-val">${value}</span>
                </button>`;
      }).join('');

      html.push(`<details class="stat-group" data-group="${group.key}"${open ? ' open' : ''}>
          <summary>${group.label}<span class="stat-group-count">${present.length}</span></summary>
          <div class="stat-group-body">${body}</div>
        </details>`);
    }
    container.innerHTML = html.join('');

    for (const el of container.querySelectorAll('.stat')) {
      el.addEventListener('click', () => StatDetail.open(el.dataset.stat, el.dataset.label));
    }

    // Opening Structure is what asks for those statistics, so redraw the strip
    // once they can be computed. Closing it costs nothing and needs no redraw.
    if (!structureOpen) {
      for (const key of heavyGroups) {
        const group = container.querySelector(`.stat-group[data-group="${key}"]`);
        if (!group) continue;
        group.addEventListener('toggle', () => {
          if (group.open) this.updateStats();
        }, { once: true });
      }
    }
  },

  /**
   * Pair two run statistics into a path through time.
   *
   * The two are not always recorded together. Most are written on both phases,
   * some only on the reproduction phase — births, and what was invested —
   * and some only on the game phase — what flowed, who revolted. Rather than
   * keeping a table of which is which, the pairing is decided from the data:
   *
   *   Where both have a value on the same frame, that frame is one point. This
   *   covers everything recorded on both phases, and any two that share a
   *   phase, at the full resolution the series holds.
   *
   *   Where they never once appear together — one reproduction-only, the other
   *   game-only — the iteration is the unit instead, taking each from whichever
   *   of its two phases recorded it. One point per iteration, which is the
   *   finest honest pairing available: the two really did happen at the same
   *   time, just not in the same half of it.
   *
   * The chart says which of the two it used, since it changes what a point
   * means.
   */
  trajectoryPoints(payload) {
    if (!payload || !payload.series) return { message: 'history is still loading' };

    const s = payload.series;
    const xKey = this.settings.trajX, yKey = this.settings.trajY;
    const xs = s[xKey], ys = s[yKey];
    const phases = s.phase || [], iterations = s.iteration || [];
    if (!xs || !ys) return { message: 'this run has no history for one of these' };

    const usable = v => v !== null && v !== undefined && Number.isFinite(v);

    const sameFrame = [];
    for (let i = 0; i < xs.length; i++) {
      if (!this.framePassesFilter(phases[i])) continue;
      if (!usable(xs[i]) || !usable(ys[i])) continue;
      sameFrame.push({ x: xs[i], y: ys[i], t: iterations[i] });
    }
    if (sameFrame.length >= 2) {
      return { points: sameFrame, pairing: 'one point per frame' };
    }

    // Never together on a frame, so pair the phases of each iteration. The
    // phase filter is ignored here on purpose: it would leave one of the two
    // with nothing, and the whole reason to be in this branch is that they
    // live on opposite halves of an iteration.
    const byIteration = new Map();
    for (let i = 0; i < xs.length; i++) {
      const at = iterations[i];
      let slot = byIteration.get(at);
      if (!slot) { slot = { x: null, y: null, t: at }; byIteration.set(at, slot); }
      if (slot.x === null && usable(xs[i])) slot.x = xs[i];
      if (slot.y === null && usable(ys[i])) slot.y = ys[i];
    }
    const paired = [...byIteration.values()]
      .filter(p => p.x !== null && p.y !== null)
      .sort((a, b) => a.t - b.t);

    if (paired.length >= 2) {
      return { points: paired, pairing: 'one point per iteration, across its phases' };
    }
    return { message: 'these two are never recorded at the same time' };
  },

  /** Raw values for a domain-qualified metric, whichever domain it names. */
  chartValues(parsed) {
    return parsed.domain === 'edge'
      ? this.metrics.edgeValues(parsed.key)
      : this.metrics.nodeValues(parsed.key);
  },

  updateCharts() {
    if (!this.metrics) return;
    const s = this.settings;

    // First, and before anything below can return early. It used to sit after
    // the heatmap's domain check, so picking a node metric against an edge one
    // stopped the trajectory redrawing at all and its own controls went dead.
    this.updateTrajectory();

    const dist = Metrics.parse(s.distMetric);
    drawHistogram(document.getElementById('distHist'), this.chartValues(dist), {
      bins: 30, colormap: s.nodeColormap, reverse: s.nodeColorReverse,
      logScale: s.histDistX === 'log', logCount: s.histDistY === 'log',
      signed: Metrics.isSigned(dist.domain, dist.key),
      format: v => Metrics.format(dist.domain, dist.key, v)
    });

    const heat = document.getElementById('heatMap');
    const x = Metrics.parse(s.heatX), y = Metrics.parse(s.heatY);

    // A node value and an edge value describe different things, and there is
    // no correspondence between the two lists to pair them by. Say so rather
    // than plotting a grid that would mean nothing.
    if (x.domain !== y.domain) {
      drawHeatmap(heat, null, null, {
        message: 'Pick two node metrics or two edge metrics — mixing them has no pairing.'
      });
      return;
    }

    drawHeatmap(heat, this.chartValues(x), this.chartValues(y), {
      colormap: s.nodeColormap, reverse: s.nodeColorReverse,
      logX: s.histHeatX === 'log', logY: s.histHeatY === 'log',
      logCount: s.histHeatCount === 'log',
      signedX: Metrics.isSigned(x.domain, x.key),
      signedY: Metrics.isSigned(y.domain, y.key),
      formatX: v => Metrics.format(x.domain, x.key, v),
      formatY: v => Metrics.format(y.domain, y.key, v)
    });
  },

  /**
   * Curvature, phrased as what it means rather than as a bare number: whether
   * this agent sits below or above the neighbourhood it is wired into.
   */
  curvatureRow(d) {
    const v = Math.round(d.curvature);
    if (v === 0) return `Curvature 0 <span class="hint">(level with its neighbours)</span>`;
    const perNeighbour = d.degree ? Math.round(d.curvature / d.degree) : 0;
    const sense = v > 0
      ? `poorer than its neighbours by ${formatNumber(Math.abs(perNeighbour))} each`
      : `richer than its neighbours by ${formatNumber(Math.abs(perNeighbour))} each`;
    const cls = v > 0 ? 'bad' : 'good';
    return `Curvature <span class="${cls}">${v > 0 ? '+' : ''}${formatNumber(v)}</span>` +
           ` <span class="hint">(${sense})</span>`;
  },

  /**
   * Redraw the trajectory, fetching the run's history the first time it is
   * needed rather than on every frame step.
   */
  updateTrajectory() {
    const canvas = document.getElementById('trajectory');
    if (!canvas || !this.runId) return;
    const s = this.settings;
    const button = document.getElementById('btnTrajLoad');
    const payload = StatDetail.seriesCache.get(this.runId);

    if (button) button.style.display = payload ? 'none' : '';

    // Summarising a run means reading every frame it recorded, which on a
    // long run of large graphs takes minutes. Doing that unasked, every time
    // a run is opened, would be a poor trade for a chart the reader may not
    // want, so it waits to be asked and is then kept for the session.
    if (!payload) {
      drawTrajectory(canvas, null, {
        message: this._trajectoryLoading
          ? 'reading every recorded frame\u2026'
          : 'press Load history to summarise this run'
      });
      return;
    }

    const result = this.trajectoryPoints(payload);
    drawTrajectory(canvas, result.points, {
      colormap: s.nodeColormap, reverse: s.nodeColorReverse,
      logX: s.histTrajX === 'log', logY: s.histTrajY === 'log',
      xLabel: this.STAT_LABELS[s.trajX] || s.trajX,
      yLabel: this.STAT_LABELS[s.trajY] || s.trajY,
      footer: result.pairing || '',
      message: result.message || null
    });
  },

  showHover(i, x, y) {
    if (i < 0) { this.hoverCard.classList.add('hidden'); return; }
    const d = this.metrics.nodeDetail(i);

    const rows = [
      `<b>Node ${d.id}</b> <span class="hint">#${d.rank} by wealth</span>`,
      `Tokens ${formatNumber(d.tokens)} <span class="hint">(${(d.tokenShare * 100).toFixed(2)}% of world)</span>`,
      `Degree ${d.degree}`,
      this.curvatureRow(d),
      d.hasDelta ? this.deltaRow(d) : '<span class="hint">token change not recorded</span>',
      `Brain ${d.brainId} <span class="hint">from ${d.parentBrainId}</span>`,
      `Spawned by ${d.spawnedBy !== null ? d.spawnedBy : '—'}`
    ];

    // What this agent did depends on which phase produced the frame.
    if (d.phase === 1) {
      rows.push('<hr>');
      if (d.newbornOf !== undefined) {
        rows.push(`<b class="good">Born this phase</b> from ${d.newbornOf}`);
      }
      if (d.reproduced === null) {
        rows.push('<span class="hint">decisions not recorded</span>');
      } else if (d.reproduced) {
        rows.push(`<b class="good">Reproduced: yes</b>`);
        rows.push(`Invested ${formatNumber(d.invested)}` +
                  (d.investedShare !== null ? ` <span class="hint">(${(d.investedShare * 100).toFixed(1)}% of its tokens)</span>` : ''));
        rows.push(`Child ${d.child} · ${d.childLinks} link${d.childLinks === 1 ? '' : 's'}`);
        if (d.handedOver !== null && d.handedOver !== undefined) {
          rows.push(`Handed over ${d.handedOver} connection${d.handedOver === 1 ? '' : 's'}`);
        }
      } else {
        rows.push('Reproduced: no');
      }
    } else {
      rows.push('<hr>');
      if (d.allocated === undefined) {
        rows.push('<span class="hint">decisions not recorded</span>');
      } else {
        rows.push(`Allocated ${formatNumber(d.allocated)} <span class="hint">(${d.doctrine})</span>`);
        rows.push(`Kept at home ${formatNumber(d.keptAtHome)}` +
                  (d.allocated ? ` <span class="hint">(${((d.keptAtHome / d.allocated) * 100).toFixed(0)}%)</span>` : ''));
        rows.push(`Revolt tokens ${formatNumber(d.revolted)}`);
      }

      if (d.heldHome === undefined) {
        rows.push('<span class="hint">no bids on this node</span>');
      } else if (d.heldHome) {
        rows.push(`<b class="good">Held its own node</b> <span class="hint">(bid ${formatNumber(d.winningBid)})</span>`);
      } else {
        rows.push(`<b class="bad">Taken by ${d.takenBy}</b> <span class="hint">(bid ${formatNumber(d.winningBid)})</span>`);
      }
      if (d.wonByRevolt) rows.push('<b class="warnText">Decided by revolution</b>');
      if (d.nodesWon) rows.push(`Won ${d.nodesWon} node${d.nodesWon === 1 ? '' : 's'} this phase`);
    }

    this.hoverCard.innerHTML = rows.join('<br>').replace(/<br><hr><br>/g, '<hr>');

    // Flip the card when it would otherwise run off the canvas.
    const wrap = this.canvas.getBoundingClientRect();
    this.hoverCard.style.left = '0px';
    this.hoverCard.style.top = '0px';
    this.hoverCard.classList.remove('hidden');
    const card = this.hoverCard.getBoundingClientRect();

    const left = (x + 14 + card.width > wrap.width) ? x - card.width - 14 : x + 14;
    const top = (y + 14 + card.height > wrap.height) ? y - card.height - 14 : y + 14;
    this.hoverCard.style.left = `${Math.max(4, left)}px`;
    this.hoverCard.style.top = `${Math.max(4, top)}px`;
  },

  /** How this node's pile moved across the phase, phrased for the reader. */
  deltaRow(d) {
    if (d.delta > 0) {
      const born = d.phase === 1 && d.newbornOf !== undefined;
      return `<b class="good">+${formatNumber(d.delta)} tokens</b>` +
             (born ? ' <span class="hint">(endowment at birth)</span>' : '');
    }
    if (d.delta < 0) {
      return `<b class="bad">${formatNumber(d.delta)} tokens</b>`;
    }
    return 'No token change';
  }

});
