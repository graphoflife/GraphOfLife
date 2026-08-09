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
/**
 * Round numbers to rule a grid at.
 *
 * A grid drawn at whatever the data happens to span gives labels like 0.0473,
 * which are noise to read. This steps up to the nearest 1, 2, 2.5 or 5 times a
 * power of ten, so the lines land on values worth putting a number against.
 */
function niceTicks(lo, hi, target = 5) {
  if (!(hi > lo)) return { ticks: [lo], step: 1 };

  const rough = (hi - lo) / Math.max(1, target);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const scaled = rough / magnitude;
  const step = magnitude *
    (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 2.5 ? 2.5 : scaled <= 5 ? 5 : 10);

  const ticks = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
    // Repeated addition drifts; snap a near-zero tick to exactly zero.
    ticks.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return { ticks, step };
}

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
    revolutions: 'Nodes taken by a coalition rather than by the single largest allocator. A revolution succeeds when the accumulated weaker allocators outweigh everyone above them plus the hegemon. Only recorded on runs created with revolutions enabled — an absent value means the rule was off and nodes simply went to whoever allocated the most, which is not the same as the rule being on and never firing.',
    starved: 'Agents removed for holding zero tokens.',
    orphaned: 'Agents removed for being outside the largest connected component. They may have been perfectly wealthy — they were simply cut off.',
    loops: 'How many of the graph\'s independent loops pass through this element. Counting every loop through a node is intractable, so this counts a basis: take a spanning tree and each remaining edge closes exactly one loop, and those loops generate all the others. There are exactly as many as the Loops total, so the two are on the same footing. The tree is built breadth-first to keep the loops short and local; a different tree would give a different basis, so read it as a fair sample of the loop structure rather than a canonical count.',
    cycleRank: 'How many independent loops the graph contains. Counting loops one by one is hopeless, since their number grows exponentially, but the independent count is exact: take a spanning tree and every edge left over closes exactly one new loop, giving edges minus nodes plus components. Zero means the graph is a tree with no loops at all.',
    loopDensity: 'Independent loops as a share of edges — the fraction of connections that are redundant in the sense that removing one would not disconnect anything. Low values mean a spindly, tree-like graph; high values mean a densely interwoven one.',
    bridges: 'Edges that lie on no loop at all. Removing a bridge splits the graph in two, so these are the connections holding otherwise separate regions together, and losing one to pruning can cut a whole branch adrift.',
    triangles: 'Closed triples of mutually connected agents — the shortest loop there is. Unlike loops in general these can be counted exactly, since the triangles on an edge are simply the neighbours its two endpoints share.',
    transitivity: 'The chance that two neighbours of an agent are themselves neighbours, from 0 to 1. High values mean tight local cliques; low values mean a sparse, tree-like structure with little local redundancy.',
    dimension: 'How many dimensions the graph behaves as though it has, in the spirit of the Wolfram Physics Project: walk outward from a node and measure how fast the frontier grows. In d-dimensional space the shell reached at r steps holds about r to the power d minus one, so the slope of log shell against log radius, plus one, estimates d. Averaged over many starting points, discarding radii whose ball has already swallowed half the graph, since past that the growth is measuring the boundary rather than the geometry. Checked against lattices of known dimension it returns 1.00 for a chain, 1.92 for a square grid and 2.56 for a cubic one: exact in one dimension and increasingly conservative above it, because a few steps is not enough room for the growth to reach its asymptote. Read it as a rough index rather than a precise figure. A small-world graph has no honest dimension at all — its neighbourhoods grow exponentially rather than polynomially — which shows up here as a large or unsteady number, and that is a real finding about the graph rather than a broken measurement.',
    degreeEntropy: 'Shannon entropy of the degree distribution, in bits: how much variety there is in how connected agents are. Zero means every agent has exactly the same number of connections. Higher values mean a broader mix of hubs and leaves.',
    degreeEvenness: 'Degree entropy against the most even it could be for the same number of distinct degrees. One means every degree that occurs is equally common.',
    tokenEntropy: 'Shannon entropy of wealth, in bits, treating each agent\'s share of the tokens as a probability. It peaks at log2 of the population when everyone holds the same amount and falls as wealth concentrates — the same story the Gini coefficient tells, in the language of information rather than inequality.',
    tokenEvenness: 'Token entropy divided by its maximum, so 100 percent means perfectly equal wealth regardless of population size. Unlike the raw entropy this is comparable between frames as the population grows.',
    components: 'Connected pieces of the graph. Cleanup keeps only the largest, so this should read 1 on any recorded frame; anything else would mean the culling step had missed something.',
    density: 'Edges present as a share of every edge that could exist. Falls quickly as the population grows, since possible edges grow with the square of the node count.',
    medianDegree: 'The middle agent\'s number of connections. Well below the mean means a few hubs are carrying the average.',
    minDegree: 'Connections held by the least connected agent.',
    leaves: 'Agents with exactly one connection. They depend entirely on a single neighbour, and lose their place in the graph if that edge is pruned.',
    meanTokens: 'Total tokens divided by the number of agents. Since tokens are conserved, this moves only because the population changes.',
    minTokens: 'Wealth of the poorest surviving agent. Anything that reached zero was already removed by cleanup.',
    topDecileShare: 'Share of all tokens held by the richest tenth of agents. A blunter companion to the Gini coefficient — easier to picture, less sensitive to the middle of the distribution.',
    distinctLineages: 'How many different parent genomes are represented. Lower than the distinct brain count, since many mutated children share one parent.',
    reproTokenShare: 'Share of every token in the world that was committed to newborns this phase, counting what each parent handed over. This measures how much of the whole economy the population spent on reproducing, whereas Mean investment averages what each individual parent gave away as a fraction of its own pile — a world where a handful of poor agents each give away everything scores high on one and low on the other.',
    meanInvestedShare: 'Average share of its own tokens that a reproducing agent handed to its child. Reproduction is paid for out of the parent\'s own life, so this is how much of itself the average parent gave away.',
    handovers: 'Connections a parent gave to its newborn this phase: the edge moves from parent to child rather than being copied, so the parent ends up with one fewer. Only recorded on runs created with handover enabled — an absent value means the mechanic was off, which is not the same as it being on and never used. If the newborn was already wired to that neighbour, the two collapse into the single edge a simple graph can hold, and only the anchor changes.',
    meanChildLinks: 'Average number of connections a newborn was wired to. One means children hang off a single neighbour; higher values mean they are born well embedded.',
    totalFlow: 'Tokens that crossed an edge this phase, ignoring what agents kept on themselves. This is the volume of actual traffic in the network.',
    meanEdgeFlow: 'Average tokens carried per edge that carried anything. Edges that carried nothing are pruned at the end of the phase.',
    maxEdgeFlow: 'Tokens carried by the single busiest edge — the largest bid sent along one connection.',
    selfAllocationShare: 'Share of all allocated tokens that agents kept on their own node rather than spending on neighbours. High values mean a defensive population; low values mean an aggressive one.',
    revoltShare: 'Share of allocated tokens flagged as revolutionary. Revolution tokens count toward coalitions against the largest allocator instead of backing the strongest bid.',
    spreadShare: 'Share of agents that spread their tokens proportionally across targets rather than going all-in on a single one. The choice is made by each agent\'s own mode head, so this tracks an evolving strategy.',
    heldHomeShare: 'Share of nodes whose occupant kept the node. The rest were taken over, with the winner\'s genome copied in.',
    prunedEdges: 'Edges removed for carrying no tokens this phase. Connections have to be used to survive.',
    maxTokenAdded: 'The largest gain any single agent made this phase. In a reproduction phase that is usually a newborn receiving its endowment; in a game phase it is a node that collected a heavy bid, often one that was conquered by a wealthy neighbour. This will not cancel against Max token lost, for two reasons. First, both are maxima over individual agents: the biggest winner and the biggest loser are different agents, and nothing ties one\'s fortune to the other\'s. Second, even the totals do not cancel — see Max token lost.',
    maxTokenLost: 'The largest loss any single agent took this phase, given as a positive number. In a reproduction phase that is a parent paying for a child; in a game phase it is an agent that spent its pile on neighbours and got little back. Summed across everyone, gains exceed losses by exactly what the agents who did not survive the phase were holding when it began. Those agents are gone from the frame, so their losses are never counted, while the tokens they let go of turn up as gains for the survivors — some spent on neighbours during the phase, the rest scattered by cleanup. Tokens are still conserved overall; it is the bookkeeping that is one-sided, because a frame can only describe agents that are still alive to be described.',
    gainers: 'Agents that ended the phase with more tokens than they started it with.',
    losers: 'Agents that ended the phase with fewer tokens than they started it with.',
    redistributed: 'Tokens recovered from agents that died this phase and scattered uniformly over the survivors, keeping the global count conserved.'
  },

  // Statistics that count nodes, and so are also meaningful as a percentage
  // of the population that entered the phase.
  SHARE_KEYS: new Set(['births', 'revolutions', 'starved', 'orphaned', 'leaves',
                       'gainers', 'losers']),

  init() {
    this.el = document.getElementById('statDetail');
    this.titleEl = document.getElementById('statDetailTitle');
    this.textEl = document.getElementById('statDetailText');
    this.footEl = document.getElementById('statDetailFoot');
    this.canvas = document.getElementById('statDetailChart');
    this.progressEl = document.getElementById('statDetailProgress');
    this.progressFill = document.getElementById('statDetailProgressFill');
    this.progressLabel = document.getElementById('statDetailProgressLabel');

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
    this.el.classList.remove('hidden');

    const runId = Viewer.runId;
    const cached = this.seriesCache.has(runId);
    if (!cached) this.showProgress();

    try {
      this.series = await this.load(runId);
    } catch (err) {
      this.hideProgress();
      this.footEl.textContent = `Could not load history: ${err.message}`;
      return;
    }
    this.hideProgress();
    this.redraw();
  },

  async load(runId) {
    if (this.seriesCache.has(runId)) return this.seriesCache.get(runId);

    // Watch the build while it runs. Changing a statistic invalidates the
    // cache, and re-reading every frame of a long run takes long enough that a
    // blank wait is worse than useless.
    const watching = this.watchProgress(runId);
    try {
      const payload = await API.getSeries(runId);
      this.seriesCache.set(runId, payload);
      return payload;
    } finally {
      watching.stop();
    }
  },

  // ---- progress ------------------------------------------------------

  showProgress() {
    this.footEl.textContent = '';
    this.progressEl.classList.remove('hidden');
    this.setProgress(null, 0, 0);
  },

  hideProgress() {
    this.progressEl.classList.add('hidden');
  },

  /**
   * Move the bar.
   *
   * A null fraction means the server has not said how much there is to do yet,
   * so the bar sweeps rather than claiming a position it cannot know.
   */
  setProgress(fraction, done, total) {
    if (fraction === null) {
      this.progressFill.classList.add('indeterminate');
      this.progressFill.style.width = '';
      this.progressLabel.textContent = 'Analysing frames…';
      return;
    }
    this.progressFill.classList.remove('indeterminate');
    this.progressFill.style.width = `${Math.round(fraction * 100)}%`;
    this.progressLabel.textContent =
      `Analysing frames… ${formatNumber(done)} of ${formatNumber(total)} (${Math.round(fraction * 100)}%)`;
  },

  /** Poll the server for build progress until told to stop. */
  watchProgress(runId) {
    let stopped = false;
    const tick = async () => {
      if (stopped) return;
      try {
        const p = await API.getSeriesProgress(runId);
        if (!stopped) {
          if (p.building && p.total > 0) this.setProgress(p.done / p.total, p.done, p.total);
          else this.setProgress(null, 0, 0);
        }
      } catch (err) {
        // A failed poll only costs the bar its accuracy, never the load.
      }
      if (!stopped) setTimeout(tick, 250);
    };
    setTimeout(tick, 120);
    return { stop() { stopped = true; } };
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

    const padL = 62, padR = 12, padT = 10, padB = 30;
    const plotW = w - padL - padR, plotH = h - padT - padB;

    let lo = Math.min(...ys), hi = Math.max(...ys);
    if (hi - lo < 1e-12) { lo -= 0.5; hi += 0.5; }

    // Widen to the round numbers, so the top and bottom lines are labelled
    // values rather than wherever the data happened to stop.
    const yGrid = niceTicks(lo, hi, 5);
    if (yGrid.ticks.length > 1) {
      lo = Math.min(lo, yGrid.ticks[0]);
      hi = Math.max(hi, yGrid.ticks[yGrid.ticks.length - 1]);
    }

    const xLo = xs[0], xHi = xs[xs.length - 1];
    const xGrid = niceTicks(xLo, xHi, 5);

    const xAt = v => padL + (xHi === xLo ? plotW / 2 : ((v - xLo) / (xHi - xLo)) * plotW);
    const yAt = v => padT + plotH - ((v - lo) / (hi - lo)) * plotH;

    const fmtY = v => asShare
      ? `${+v.toFixed(2)}%`
      : (Math.abs(v) >= 1000 ? Math.round(v).toLocaleString('en-US')
                             : String(+v.toFixed(Math.abs(v) < 1 ? 3 : 2)));

    ctx.font = '10px system-ui, sans-serif';
    ctx.lineWidth = 1;

    // Horizontal grid
    for (const v of yGrid.ticks) {
      const y = Math.round(yAt(v)) + 0.5;
      if (y < padT - 1 || y > padT + plotH + 1) continue;
      ctx.strokeStyle = '#1e2733';
      ctx.beginPath();
      ctx.moveTo(padL, y);
      ctx.lineTo(padL + plotW, y);
      ctx.stroke();

      ctx.fillStyle = '#8fa3b5';
      const label = fmtY(v);
      ctx.fillText(label, padL - 6 - ctx.measureText(label).width, y + 3);
    }

    // Vertical grid
    for (const v of xGrid.ticks) {
      const x = Math.round(xAt(v)) + 0.5;
      if (x < padL - 1 || x > padL + plotW + 1) continue;
      ctx.strokeStyle = '#1e2733';
      ctx.beginPath();
      ctx.moveTo(x, padT);
      ctx.lineTo(x, padT + plotH);
      ctx.stroke();

      ctx.fillStyle = '#8fa3b5';
      const label = Math.round(v).toLocaleString('en-US');
      ctx.fillText(label, x - ctx.measureText(label).width / 2, h - 12);
    }

    // Axes, a shade brighter than the grid
    ctx.strokeStyle = '#33404f';
    ctx.beginPath();
    ctx.moveTo(padL + 0.5, padT);
    ctx.lineTo(padL + 0.5, padT + plotH + 0.5);
    ctx.lineTo(padL + plotW, padT + plotH + 0.5);
    ctx.stroke();

    // Where the frame on screen sits, so the number in the strip has a home
    const currentIteration = Viewer.frame ? Viewer.frame.iteration : null;
    if (currentIteration !== null && currentIteration >= xLo && currentIteration <= xHi) {
      ctx.strokeStyle = '#4fb3ff';
      ctx.globalAlpha = 0.45;
      ctx.beginPath();
      ctx.moveTo(xAt(currentIteration), padT);
      ctx.lineTo(xAt(currentIteration), padT + plotH);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // The curve
    ctx.strokeStyle = '#4fb3ff';
    ctx.lineWidth = 1.6;
    ctx.beginPath();
    for (let i = 0; i < ys.length; i++) {
      const x = xAt(xs[i]), y = yAt(ys[i]);
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();

    ctx.fillStyle = '#5b6b7c';
    ctx.font = '9px system-ui, sans-serif';
    const axisLabel = 'iteration';
    ctx.fillText(axisLabel, padL + plotW - ctx.measureText(axisLabel).width, h - 1);

    const sampled = this.series && this.series.sampled;
    this.footEl.textContent =
      `${formatNumber(ys.length)} point${ys.length === 1 ? '' : 's'} · ${Viewer.phaseFilterLabel()}` +
      (asShare ? ' · shown as a share of the nodes that entered the phase' : '') +
      (sampled ? ` · sampled every ${formatNumber(this.series.stride)} iterations` : '');
  }
};
