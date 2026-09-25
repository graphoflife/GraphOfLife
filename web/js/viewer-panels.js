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
    cutRisk: 'Worst cut',
    cutRiskBefore: 'Worst cut, pre-cull',
    coreShare: 'Core share',
    spectralGap: 'Spectral gap \u03bb\u2082',
    lightningScore: 'Lightning score',
    cyclingShare: 'Tokens circulating',
    lightningLongest: 'Longest lightning',
    flowImbalance: 'Flow imbalance',
    netLightningScore: 'Net lightning score',
    netCyclingShare: 'Net tokens circulating',
    netLightningLongest: 'Longest net lightning',
    netFlowShare: 'Flow surviving cancellation',
    triangles: 'Triangles',
    transitivity: 'Clustering',
    dimension: 'Dimension',
    ricciCurvature: 'Curvature',
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
    gifts: 'Gifts',
    giftTokens: 'Gifted tokens',
    giftShare: 'Gifted share',
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

  /**
   * Which category each statistic belongs to, and the order within it.
   *
   * Phase-specific groups simply come out empty on the other phase, so the
   * Reproduction section disappears on a game frame rather than showing a row
   * of dashes.
   */
  STAT_GROUPS: [
    { key: 'general', label: 'General', open: true, keys: [
      'nodes', 'edges', 'tokens', 'meanTokens', 'medianTokens', 'maxTokens', 'minTokens',
      'gini', 'topDecileShare', 'tokenEntropy', 'tokenEvenness',
      'maxTokenAdded', 'maxTokenLost', 'gainers', 'losers',
      'starved', 'orphaned', 'redistributed', 'cutRiskBefore',
      'distinctBrains', 'brainDiversity', 'distinctParents'
    ] },
    { key: 'reproduction', label: 'Reproduction', open: true, keys: [
      'births', 'reproTokenShare', 'meanInvestedShare', 'meanChildLinks',
      'handovers', 'gifts', 'giftTokens', 'giftShare'
    ] },
    // The lightning readings belong here rather than under General: they are
    // measured on the token flow a Blotto phase allocates, they are blank on a
    // reproduction frame, and they are read against the traffic figures they
    // sit beside.
    { key: 'blotto', label: 'Game (Blotto)', open: true, keys: [
      'totalFlow', 'meanEdgeFlow', 'maxEdgeFlow', 'selfAllocationShare',
      'revoltShare', 'spreadShare', 'revolutions', 'heldHomeShare', 'prunedEdges',
      'lightningScore', 'cyclingShare', 'lightningLongest', 'flowImbalance',
      'netLightningScore', 'netCyclingShare', 'netLightningLongest', 'netFlowShare'
    ] },
    { key: 'structure', label: 'Structure', open: false, keys: [
      'density', 'meanDegree', 'medianDegree', 'maxDegree', 'minDegree', 'leaves',
      'radius', 'diameter', 'meanPathLength',
      'cycleRank', 'loopDensity', 'bridges', 'cutRisk', 'coreShare', 'spectralGap',
      'triangles', 'transitivity', 'dimension', 'ricciCurvature',
      'degreeEntropy', 'degreeEvenness', 'components'
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
    const isOpen = (key) => {
      const el = container.querySelector(`.stat-group[data-group="${key}"]`);
      if (el) return el.open;
      const declared = this.STAT_GROUPS.find(g => g.key === key);
      return Boolean(declared && declared.open);
    };
    const heavyGroups = ['structure', 'powerlaws'];
    const structureOpen = heavyGroups.some(isOpen);
    // The lightning search is its own cost and its own group. It walks this
    // phase's allocations rather than the graph, so it is asked for on its own
    // terms — otherwise the group holding it would sit open and empty while
    // Structure was closed.
    const s = this.metrics.summary(structureOpen, isOpen('blotto'));

    // Node counts are also given as a share of the population that entered the
    // phase — "40 births" reads very differently at 100 agents than at 4,000.
    const base = s.nodesBefore || s.nodes || 0;
    const withShare = v => (base && v !== null && v !== undefined)
      ? `${formatNumber(v)} <i>${((v / base) * 100).toFixed(1)}%</i>` : formatNumber(v);

    const int = v => (v === null || v === undefined) ? '\u2014' : formatNumber(Math.round(v));
    const pct = v => (v === null || v === undefined) ? '\u2014' : `${(v * 100).toFixed(1)}%`;
    const dec = (v, n = 2) => (v === null || v === undefined) ? '\u2014' : v.toFixed(n);

    // The formatted value of every statistic this frame has. Its label is
    // looked up where it is drawn rather than carried beside it in every row.
    const cells = {
      nodes: formatNumber(s.nodes),
      edges: formatNumber(s.edges),
      tokens: formatNumber(s.tokens),
      meanTokens: int(s.meanTokens),
      medianTokens: int(s.medianTokens),
      maxTokens: formatNumber(s.maxTokens),
      minTokens: formatNumber(s.minTokens),
      gini: dec(s.gini, 3),
      topDecileShare: pct(s.topDecileShare),
      tokenEntropy: `${dec(s.tokenEntropy)} bits`,
      tokenEvenness: pct(s.tokenEvenness),
      maxTokenAdded: `+${formatNumber(s.maxTokenAdded)}`,
      maxTokenLost: `-${formatNumber(s.maxTokenLost)}`,
      gainers: formatNumber(s.gainers),
      losers: formatNumber(s.losers),
      distinctBrains: formatNumber(s.distinctBrains),
      brainDiversity: pct(s.brainDiversity),
      distinctParents: formatNumber(s.distinctParents),

      density: `${(s.density * 100).toFixed(2)}%`,
      meanDegree: dec(s.meanDegree),
      medianDegree: dec(s.medianDegree, 1),
      maxDegree: formatNumber(s.maxDegree),
      minDegree: formatNumber(s.minDegree),
      leaves: formatNumber(s.leaves),
      degreeEntropy: `${dec(s.degreeEntropy)} bits`,
      degreeEvenness: pct(s.degreeEvenness)
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
      cells[key] = exponent(s[key]);
      cells[key + 'R2'] = pct(s[key + 'R2']);
    }
    cells.assortativity = dec(s.assortativity, 3);
    cells.degreeGamma = dec(s.degreeGamma);
    cells.degreeGammaR2 = pct(s.degreeGammaR2);
    cells.degreeKMin = int(s.degreeKMin);
    cells.degreeTailShare = pct(s.degreeTailShare);
    cells.degreeGammaKS = dec(s.degreeGammaKS, 3);
    cells.boxDimension = dec(s.boxDimension);
    cells.boxDimensionR2 = pct(s.boxDimensionR2);

    // Only computed while one of the heavy groups is open, since walking the
    // whole graph costs more than the rest of this strip together.
    if (structureOpen) {
      cells.cycleRank = formatNumber(s.cycleRank);
      cells.loopDensity = pct(s.loopDensity);
      cells.bridges = formatNumber(s.bridges);
      cells.cutRisk = pct(s.cutRisk);
      cells.coreShare = pct(s.coreShare);
      cells.spectralGap = dec(s.spectralGap, 4);
      cells.triangles = formatNumber(s.triangles);
      cells.transitivity = dec(s.transitivity, 3);
      cells.dimension = dec(s.dimension);
      cells.ricciCurvature = dec(s.ricciCurvature, 3);
      cells.radius = formatNumber(s.radius);
      cells.diameter = formatNumber(s.diameter);
      cells.meanPathLength = dec(s.meanPathLength);
      cells.components = formatNumber(s.components);

    }

    // Present only when the phase produced them.
    if (s.births !== null) {
      cells.births = formatNumber(s.births);
      cells.reproTokenShare = pct(s.reproTokenShare);
      cells.meanInvestedShare = pct(s.meanInvestedShare);
      cells.meanChildLinks = dec(s.meanChildLinks);
      if (s.handovers !== null) cells.handovers = formatNumber(s.handovers);
    }
    // Gifts, on their own condition rather than on `births`: an agent with
    // nothing to spare for a child may still give a neighbour a token, so a
    // phase can have gifts and no births at all. Absent on a run without the
    // mechanic, which is why this asks rather than assuming.
    if (s.gifts !== null && s.gifts !== undefined) {
      cells.gifts = formatNumber(s.gifts);
      cells.giftTokens = formatNumber(s.giftTokens);
      cells.giftShare = pct(s.giftShare);
    }
    if (s.totalFlow !== null) {
      cells.totalFlow = formatNumber(s.totalFlow);
      cells.meanEdgeFlow = dec(s.meanEdgeFlow, 1);
      cells.maxEdgeFlow = formatNumber(s.maxEdgeFlow);
      cells.selfAllocationShare = pct(s.selfAllocationShare);
      cells.spreadShare = pct(s.spreadShare);
      // Null when the run has revolutions off. Formatting that as 0% would
      // claim nobody revolted, when in fact nobody could.
      if (s.revoltShare !== null) cells.revoltShare = pct(s.revoltShare);
    }
    if (s.revolutions !== null) cells.revolutions = formatNumber(s.revolutions);
    if (s.heldHomeShare !== null) cells.heldHomeShare = pct(s.heldHomeShare);
    if (s.prunedEdges !== null) cells.prunedEdges = formatNumber(s.prunedEdges);
    if (s.starved !== null) cells.starved = formatNumber(s.starved);
    if (s.orphaned !== null) cells.orphaned = formatNumber(s.orphaned);
    if (s.cutRiskBefore !== null && s.cutRiskBefore !== undefined) {
      cells.cutRiskBefore = pct(s.cutRiskBefore);
    }
    // Only a game phase moves tokens across links, so these are absent on a
    // reproduction frame rather than zero.
    if (s.lightningScore !== null && s.lightningScore !== undefined) {
      cells.lightningScore = formatNumber(s.lightningScore);
      cells.cyclingShare = pct(s.cyclingShare);
      cells.lightningLongest = formatNumber(s.lightningLongest);
      cells.flowImbalance = pct(s.flowImbalance);
      cells.netLightningScore = formatNumber(s.netLightningScore);
      cells.netCyclingShare = pct(s.netCyclingShare);
      cells.netLightningLongest = formatNumber(s.netLightningLongest);
      cells.netFlowShare = pct(s.netFlowShare);
    }
    if (s.redistributed !== null) cells.redistributed = formatNumber(s.redistributed);

    // Counts of agents also read as a share of those who entered the phase.
    for (const key of Metrics.POPULATION_COUNTS) {
      if (key in cells) cells[key] = withShare(s[key]);
    }

    // Remember which sections were open, so redrawing a frame does not fold
    // everything back up under the reader.
    const wasOpen = new Map();
    for (const el of container.querySelectorAll('.stat-group')) {
      wasOpen.set(el.dataset.group, el.open);
    }

    const html = [];
    for (const group of this.STAT_GROUPS) {
      const present = group.keys.filter(k => k in cells);
      if (!present.length) continue;

      const open = wasOpen.has(group.key) ? wasOpen.get(group.key) : group.open;
      const body = present.map(k => {
        const label = this.STAT_LABELS[k];
        const value = cells[k];
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

  /** The trajectory's two statistics as a path through time: see SeriesLoad.pairs. */
  trajectoryPoints(payload) {
    if (!payload || !payload.series) return { message: 'history is still loading' };
    const found = SeriesLoad.pairs(payload.series, this.settings.trajX, this.settings.trajY,
                                   phase => this.framePassesFilter(phase));
    if (!found) return { message: 'this run has no history for one of these' };
    if (!found.pairing) return { message: 'these two are never recorded at the same time' };
    return found;
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
   * Bring the run's history up to what these statistics need, redrawing the
   * trajectory and the open statistic as it climbs and once more when it
   * ends, however it ends.
   *
   * One load, the Viewer's, for both of them. It was the stat popup's, and
   * the trajectory borrowed it, so each could cancel the other: opening a
   * statistic partway through the trajectory's load of bridge counts stopped
   * it, and the Load history button came back. Now what is already there is
   * not asked for, what is already being loaded is left to arrive, and
   * anything else widens the load in flight to cover both.
   *
   * Only as deep as they need, so a population curve does not wait on bridge
   * counts. Never rejects: a failure is said under the open statistic, and
   * the trajectory offers its button again.
   */
  loadHistory(keys) {
    const runId = this.runId;
    if (!runId || SeriesLoad.ready(runId, keys, this.frameCount)) return Promise.resolve();
    const loading = Jobs.busy(this) ? this._history.keys : [];
    if (loading.length && SeriesLoad.covers(runId, loading, keys)) return this._history.done;

    const wanted = [...new Set([...loading, ...keys])];
    const drawn = () => {
      if (this.runId !== runId) return;
      StatDetail.refresh();
      this.updateTrajectory();
    };
    let failure = null;
    const done = Jobs.run(this, 'Summarising the run',
      job => SeriesLoad.climb(runId, wanted, { job, onStep: drawn }),
      err => { failure = err; })
      .then(() => {
        drawn();
        // After the last draw, which would otherwise write over it.
        if (failure) StatDetail.failed(failure);
      });
    this._history = { keys: wanted, done };
    return done;
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
    const payload = SeriesLoad.cache.get(this.runId);
    const loading = Jobs.busy(this);

    // Offered whenever the history does not yet answer these two axes — never
    // loaded, cut short, or loaded without the graph statistics one of them
    // plots. Hiding it as soon as anything was loaded left a history cut short
    // with no way to finish it.
    const ready = SeriesLoad.ready(this.runId, [s.trajX, s.trajY], this.frameCount);
    if (button) button.style.display = ready || loading ? 'none' : '';

    // Summarising a run means reading every frame it recorded, which on a
    // long run of large graphs takes minutes. Doing that unasked, every time
    // a run is opened, would be a poor trade for a chart the reader may not
    // want, so it waits to be asked and is then kept for the session.
    if (!payload) {
      drawTrajectory(canvas, null, {
        message: loading
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
