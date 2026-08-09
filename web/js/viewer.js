/*
 * The Viewer tab: step through a recorded run and control how it is drawn.
 *
 * Frames are fetched lazily and cached, the force layout runs continuously in
 * an animation loop, and every visualization control writes into one settings
 * object that the renderer reads.
 */
const Viewer = {
  runId: null,
  meta: null,
  frameCount: 0,      // frames on disk
  visible: [],        // frame indices passing the phase filter
  position: 0,        // index into `visible`
  frameIndex: 0,      // the actual frame index being shown
  frame: null,
  metrics: null,
  cache: new Map(),
  playing: false,
  playAccumulator: 0,
  lastTime: 0,
  phaseFilter: 'all',

  settings: {
    nodeColorBy: 'tokens', nodeColormap: 'viridis', nodeColorReverse: false,
    nodeSizeBy: 'tokens', nodeSizeMin: 1.5, nodeSizeMax: 9,
    nodeAlpha: 0.9, nodeOutline: false,
    edgeShow: true, edgeColorBy: 'constant', edgeColormap: 'cividis',
    edgeFlatColor: '#5f7d95', edgeWidthBy: 'constant',
    edgeWidthMin: 0.3, edgeWidthMax: 1.6, edgeAlpha: 0.25,
    bgStyle: 'solid', bgColorA: '#0d1117', bgColorB: '#1d2530',
    showLegend: true, layoutCarry: true,

    // Layout lives in the settings too, so a preset restores the whole look
    // including how the graph arranges itself, not just its colours.
    forceCharge: 20, forceLink: 0.12, forceCenter: 0.012,
    forceAngular: 0.15, forceDamping: 0.86, forceTheta: 1.2,
    dimensions: 3, autoFit: true,

    // Histogram axis scales, per chart and per axis.
    histDegreeX: 'linear', histDegreeY: 'linear',
    histTokenX: 'log', histTokenY: 'linear'
  },

  init() {
    this.canvas = document.getElementById('graphCanvas');
    this.renderer = new GraphRenderer(this.canvas);
    // The layout runs in a worker when one is available, so a slow tick on a
    // large graph cannot freeze the interface.
    this.layout = new LayoutClient();
    this.emptyEl = document.getElementById('canvasEmpty');
    this.hoverCard = document.getElementById('hoverCard');

    populateColormapSelect(document.getElementById('nodeColormap'), this.settings.nodeColormap);
    populateColormapSelect(document.getElementById('edgeColormap'), this.settings.edgeColormap);

    this.bindControls();
    this.bindPlayback();
    this.bindCanvas();
    this.bindToggles();
    this.bindPresets();
    this.bindAxisToggles();
    this.refreshPresetList();

    // The declared defaults are only values until something applies them, so
    // push them into the layout, the renderer and the controls up front.
    this.applyLayoutSettings();
    this.syncControlsFromSettings();
    this.setDimensions(this.settings.dimensions);
    this.setAutoFit(this.settings.autoFit);

    if (window.ResizeObserver) {
      this._observer = new ResizeObserver(() => this.resize());
      this._observer.observe(this.canvas.parentElement);
    }
    window.addEventListener('resize', () => this.resize());

    this.resize();
    requestAnimationFrame(t => this.animate(t));
  },

  // ------------------------------------------------------------------
  // Wiring
  // ------------------------------------------------------------------

  bindControls() {
    const bind = (id, key, transform = v => v, needsMetrics = true) => {
      const el = document.getElementById(id);
      if (!el) return;
      const event = (el.type === 'checkbox' || el.tagName === 'SELECT') ? 'change' : 'input';
      el.addEventListener(event, () => {
        this.settings[key] = transform(el.type === 'checkbox' ? el.checked : el.value);
        if (needsMetrics) this.rebuildMetrics();
        this.updateCharts();
      });
    };

    const num = v => Number(v);

    bind('nodeColorBy', 'nodeColorBy');
    bind('nodeColormap', 'nodeColormap');
    bind('nodeColorReverse', 'nodeColorReverse');
    bind('nodeSizeBy', 'nodeSizeBy');
    bind('nodeSizeMin', 'nodeSizeMin', num, false);
    bind('nodeSizeMax', 'nodeSizeMax', num, false);
    bind('nodeAlpha', 'nodeAlpha', num, false);
    bind('nodeOutline', 'nodeOutline', v => v, false);

    bind('edgeShow', 'edgeShow', v => v, false);
    bind('edgeColorBy', 'edgeColorBy');
    bind('edgeColormap', 'edgeColormap', v => v, false);
    bind('edgeFlatColor', 'edgeFlatColor', v => v, false);
    bind('edgeWidthBy', 'edgeWidthBy');
    bind('edgeWidthMin', 'edgeWidthMin', num, false);
    bind('edgeWidthMax', 'edgeWidthMax', num, false);
    bind('edgeAlpha', 'edgeAlpha', num, false);

    bind('bgStyle', 'bgStyle', v => v, false);
    bind('bgColorA', 'bgColorA', v => v, false);
    bind('bgColorB', 'bgColorB', v => v, false);
    bind('showLegend', 'showLegend', v => v, false);
    bind('layoutCarry', 'layoutCarry', v => v, false);

    // Layout sliders are ordinary settings; applyLayoutSettings pushes them
    // into the simulation so presets and the controls stay in step.
    for (const id of ['forceCharge', 'forceLink', 'forceCenter', 'forceAngular',
                      'forceDamping', 'forceTheta']) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.addEventListener('input', () => {
        this.settings[id] = Number(el.value);
        this.applyLayoutSettings();
        this.layout.reheat(0.6);
      });
    }

    document.getElementById('btnReheat').addEventListener('click', () => this.layout.reheat(1));
    document.getElementById('btnRelayout').addEventListener('click', () => this.layout.scatter());
    document.getElementById('btnFit').addEventListener('click', () => this.setAutoFit(!this.settings.autoFit));
    document.getElementById('btnReloadFrames').addEventListener('click', () => this.reload());

    document.getElementById('runPicker').addEventListener('change', e => {
      if (e.target.value) this.load(e.target.value);
    });

    for (const btn of document.querySelectorAll('[data-preset]')) {
      btn.addEventListener('click', () => this.applySettings(Presets.builtIn(btn.dataset.preset)));
    }
  },

  bindAxisToggles() {
    for (const group of document.querySelectorAll('.axis-group')) {
      const key = 'hist' + group.dataset.axis.charAt(0).toUpperCase() + group.dataset.axis.slice(1);
      for (const btn of group.querySelectorAll('.axis-btn')) {
        btn.addEventListener('click', () => {
          this.settings[key] = btn.dataset.scale;
          for (const sibling of group.querySelectorAll('.axis-btn')) {
            sibling.classList.toggle('active', sibling === btn);
          }
          this.updateCharts();
        });
      }
    }
  },

  syncAxisToggles() {
    for (const group of document.querySelectorAll('.axis-group')) {
      const key = 'hist' + group.dataset.axis.charAt(0).toUpperCase() + group.dataset.axis.slice(1);
      for (const btn of group.querySelectorAll('.axis-btn')) {
        btn.classList.toggle('active', btn.dataset.scale === this.settings[key]);
      }
    }
  },

  bindToggles() {
    for (const btn of document.querySelectorAll('#dimToggle .seg-btn')) {
      btn.addEventListener('click', () => this.setDimensions(Number(btn.dataset.dim)));
    }
    for (const btn of document.querySelectorAll('#phaseToggle .seg-btn')) {
      btn.addEventListener('click', () => this.setPhaseFilter(btn.dataset.phase));
    }
  },

  bindPresets() {
    const list = document.getElementById('savedPresets');
    const nameInput = document.getElementById('presetName');

    list.addEventListener('change', () => { nameInput.value = list.value; });
    list.addEventListener('dblclick', () => this.applySettings(Presets.get(list.value)));

    document.getElementById('btnApplyPreset').addEventListener('click', () => {
      if (list.value) this.applySettings(Presets.get(list.value));
    });

    document.getElementById('btnSavePreset').addEventListener('click', () => {
      const name = nameInput.value.trim();
      if (!name) { alert('Give the preset a name first.'); return; }
      if (Presets.get(name) && !confirm(`"${name}" already exists. Overwrite it?`)) return;
      if (Presets.put(name, this.settings)) this.refreshPresetList(name);
    });

    document.getElementById('btnUpdatePreset').addEventListener('click', () => {
      const name = list.value;
      if (!name) { alert('Select a saved preset to update.'); return; }
      if (Presets.put(name, this.settings)) this.refreshPresetList(name);
    });

    document.getElementById('btnDeletePreset').addEventListener('click', () => {
      const name = list.value;
      if (!name) { alert('Select a saved preset to delete.'); return; }
      if (!confirm(`Delete preset "${name}"?`)) return;
      if (Presets.remove(name)) { nameInput.value = ''; this.refreshPresetList(); }
    });
  },

  refreshPresetList(selected) {
    const list = document.getElementById('savedPresets');
    list.innerHTML = '';
    for (const name of Presets.names()) {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      if (name === selected) opt.selected = true;
      list.appendChild(opt);
    }
  },

  bindPlayback() {
    document.getElementById('btnFirst').addEventListener('click', () => this.goToPosition(0));
    document.getElementById('btnPrev').addEventListener('click', () => this.goToPosition(this.position - 1));
    document.getElementById('btnNext').addEventListener('click', () => this.goToPosition(this.position + 1));
    document.getElementById('btnLast').addEventListener('click', () => this.goToPosition(this.visible.length - 1));
    document.getElementById('btnPlay').addEventListener('click', () => this.togglePlay());

    const slider = document.getElementById('frameSlider');
    slider.addEventListener('input', () => this.goToPosition(Number(slider.value)));

    document.addEventListener('keydown', e => {
      if (!App.isViewerActive()) return;
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      switch (e.key) {
        case 'ArrowLeft':  this.goToPosition(this.position - (e.shiftKey ? 10 : 1)); break;
        case 'ArrowRight': this.goToPosition(this.position + (e.shiftKey ? 10 : 1)); break;
        case 'ArrowUp':    this.goToPosition(this.position - this.stride()); break;
        case 'ArrowDown':  this.goToPosition(this.position + this.stride()); break;
        case 'Home':       this.goToPosition(0); break;
        case 'End':        this.goToPosition(this.visible.length - 1); break;
        case ' ':          this.togglePlay(); break;
        default: return;
      }
      e.preventDefault();
    });
  },

  /**
   * Keep the whole graph framed until the camera is touched.
   *
   * Pressing Fit view turns this on and it stays on, refitting as the layout
   * settles and as frames change. Any pan, zoom or orbit is taken as "I want to
   * look at this myself" and switches it off.
   */
  setAutoFit(on) {
    this.settings.autoFit = Boolean(on);
    document.getElementById('btnFit').classList.toggle('active', this.settings.autoFit);

    if (this.settings.autoFit) {
      this.renderer.fitToContent(this.layout);   // aim; the camera glides there
    } else {
      this.renderer.holdCurrentView();           // stop chasing, stay put
    }
  },

  applyLayoutSettings() {
    const s = this.settings;
    this.layout.setParams({
      charge: s.forceCharge,
      linkStrength: s.forceLink,
      centerStrength: s.forceCenter,
      angularStrength: s.forceAngular,
      damping: s.forceDamping,
      theta: s.forceTheta
    });
  },

  /** How many positions make up one whole iteration under the current filter. */
  stride() {
    return this.phaseFilter === 'all' ? 2 : 1;
  },

  bindCanvas() {
    let dragging = false, rotating = false, lastX = 0, lastY = 0;

    this.canvas.addEventListener('mousedown', e => {
      // Alt-drag or the middle button orbits; plain drag always pans.
      rotating = this.renderer.mode3D && (e.altKey || e.button === 1);
      dragging = !rotating;
      lastX = e.clientX; lastY = e.clientY;
      if (e.button === 1) e.preventDefault();
    });

    window.addEventListener('mouseup', () => { dragging = rotating = false; });

    window.addEventListener('mousemove', e => {
      if (!dragging && !rotating) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;

      if (rotating) {
        // Orbiting does not fight the framing, it changes what "framed" means:
        // a graph that filled the canvas from one angle needs a different scale
        // from another. So auto-fit stays on and re-frames as you turn.
        this.renderer.rotate(dx * 0.007, dy * 0.007);
      } else {
        // Panning is a deliberate "let me look over here", which auto-fit would
        // immediately undo, so it hands control over.
        this.renderer.pan(dx, dy);
        this.setAutoFit(false);
      }
    });

    // Middle-click drag would otherwise trigger autoscroll.
    this.canvas.addEventListener('auxclick', e => { if (e.button === 1) e.preventDefault(); });

    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      this.renderer.zoomAt(e.clientX - rect.left, e.clientY - rect.top,
                           e.deltaY < 0 ? 1.12 : 1 / 1.12);
      // Grouped with panning rather than with orbiting: auto-fit owns the
      // scale, so leaving it on here would undo the zoom on the very next
      // frame and the wheel would feel dead.
      this.setAutoFit(false);
    }, { passive: false });

    this.canvas.addEventListener('mousemove', e => {
      if (!this.frame || dragging || rotating) return;
      const rect = this.canvas.getBoundingClientRect();
      const i = this.renderer.pick(this.frame, this.layout,
                                   e.clientX - rect.left, e.clientY - rect.top);
      this.showHover(i, e.clientX - rect.left, e.clientY - rect.top);
    });
    this.canvas.addEventListener('mouseleave', () => this.hoverCard.classList.add('hidden'));
  },

  // ------------------------------------------------------------------
  // Modes
  // ------------------------------------------------------------------

  setDimensions(dims) {
    this.settings.dimensions = dims;
    this.layout.setDimensions(dims);
    this.renderer.setMode3D(dims === 3);

    for (const btn of document.querySelectorAll('#dimToggle .seg-btn')) {
      btn.classList.toggle('active', Number(btn.dataset.dim) === dims);
    }
    document.getElementById('dimHint').textContent = dims === 3
      ? 'Drag to pan · alt-drag or middle-drag to orbit · scroll to zoom'
      : 'Drag to pan · scroll to zoom';

    this.renderer.fitToContent(this.layout);
  },

  /**
   * Which phases to step through.
   *
   * Frames are written two per recorded iteration, phase 1 then phase 2, so an
   * even frame index is always a reproduction phase and an odd one always a
   * game phase. That lets the filter be built without reading any frames.
   */
  setPhaseFilter(filter) {
    this.phaseFilter = filter;
    for (const btn of document.querySelectorAll('#phaseToggle .seg-btn')) {
      btn.classList.toggle('active', btn.dataset.phase === filter);
    }

    const shownFrame = this.frameIndex;
    this.rebuildVisible();

    // Stay as close to the current moment as the new filter allows.
    let position = this.visible.indexOf(shownFrame);
    if (position < 0) {
      position = this.visible.findIndex(idx => idx >= shownFrame);
      if (position < 0) position = this.visible.length - 1;
    }

    this.updateSlider();
    if (this.visible.length) this.goToPosition(Math.max(0, position), true);
    if (!StatDetail.el.classList.contains('hidden')) StatDetail.redraw();
  },

  rebuildVisible() {
    const visible = [];
    for (let i = 0; i < this.frameCount; i++) {
      if (this.framePassesFilter(this.phaseOfIndex(i))) visible.push(i);
    }
    this.visible = visible;
  },

  phaseOfIndex(index) {
    return (index % 2 === 0) ? 1 : 2;
  },

  framePassesFilter(phase) {
    if (this.phaseFilter === 'all') return true;
    return String(phase) === this.phaseFilter;
  },

  phaseFilterLabel() {
    if (this.phaseFilter === '1') return 'reproduction phases only';
    if (this.phaseFilter === '2') return 'game phases only';
    return 'all phases';
  },

  // ------------------------------------------------------------------
  // Loading
  // ------------------------------------------------------------------

  syncRunPicker(runs) {
    const picker = document.getElementById('runPicker');
    const current = this.runId;
    picker.innerHTML = '<option value="">— choose a run —</option>';

    for (const run of runs) {
      const opt = document.createElement('option');
      opt.value = run.id;
      opt.textContent = `${run.name} (${formatNumber(run.frame_count)} frames)`;
      if (run.id === current) opt.selected = true;
      picker.appendChild(opt);
    }
  },

  async load(runId) {
    const switching = runId !== this.runId;
    this.runId = runId;
    if (switching) {
      this.cache.clear();
      // Positions from the previous run mean nothing here; the next setFrame
      // is told not to carry them over.
      this._dropPositions = true;
      this.position = 0;
      this.frameIndex = 0;
    }
    StatDetail.invalidate(runId);

    try {
      this.meta = await API.getRun(runId);
    } catch (err) {
      this.emptyEl.textContent = `Could not load run: ${err.message}`;
      this.emptyEl.style.display = '';
      return;
    }

    this.frameCount = this.meta.frame_count || 0;
    this.rebuildVisible();

    document.getElementById('activeRunLabel').textContent = this.meta.name;
    document.getElementById('runPicker').value = runId;

    if (!this.visible.length) {
      this.frame = null;
      this.emptyEl.textContent = this.frameCount
        ? 'No frames match the current phase filter.'
        : 'This run has no recorded frames yet. Start it from the Runs tab.';
      this.emptyEl.style.display = '';
      this.updateSlider();
      return;
    }

    this.emptyEl.style.display = 'none';
    await this.goToPosition(Math.min(this.position, this.visible.length - 1), true);
    this.renderer.fitToContent(this.layout, undefined, switching);
  },

  async reload() {
    if (!this.runId) return;
    this.cache.clear();
    StatDetail.invalidate(this.runId);
    await this.load(this.runId);
  },

  async fetchFrame(index) {
    if (this.cache.has(index)) return this.cache.get(index);

    const frame = await API.getFrame(this.runId, index);
    this.cache.set(index, frame);

    // Bounded cache: frames carry full topology and can be large.
    if (this.cache.size > 60) {
      this.cache.delete(this.cache.keys().next().value);
    }
    return frame;
  },

  async goToPosition(position, force = false) {
    if (!this.runId || !this.visible.length) return;
    const target = Math.max(0, Math.min(this.visible.length - 1, position));
    if (target === this.position && this.frame && !force) return;

    this.position = target;
    const index = this.visible[target];
    this.frameIndex = index;

    try {
      this.frame = await this.fetchFrame(index);
    } catch (err) {
      this.emptyEl.textContent = `Frame ${index} could not be read: ${err.message}`;
      this.emptyEl.style.display = '';
      return;
    }

    await this.ensureDelta(this.frame, index);

    this.emptyEl.style.display = 'none';
    const carry = this.settings.layoutCarry && !this._dropPositions;
    this._dropPositions = false;
    this.layout.setFrame(this.frame.ids, this.frame.edges, this.frame.parent_ids, carry);
    this.layout.reheat(this.settings.layoutCarry ? 0.35 : 1);

    this.rebuildMetrics();
    this.updateSlider();
    this.updateStats();
    this.updateCharts();
    if (!StatDetail.el.classList.contains('hidden')) StatDetail.redraw();
  },

  /**
   * Fill in per-node token change for frames recorded before the engine
   * tracked it.
   *
   * A node's balance when a phase began is simply its balance at the end of
   * the previous phase, which is the previous frame — so the same number the
   * engine writes can be recovered without re-running anything. A node absent
   * from the previous frame did not exist yet, and counts its whole balance as
   * gained, matching how the engine treats a newborn.
   *
   * Only valid when every phase was recorded: with `export_every` above one,
   * consecutive frames are further apart than a single phase and the
   * difference would span more than the phase being shown.
   */
  async ensureDelta(frame, index) {
    if (frame.delta || index <= 0) return;
    if (this.meta && this.meta.config && (this.meta.config.export_every || 1) !== 1) return;

    let previous;
    try {
      previous = await this.fetchFrame(index - 1);
    } catch (err) {
      return;   // no earlier frame to compare against; leave it unrecorded
    }

    const before = new Map();
    previous.ids.forEach((id, i) => before.set(id, previous.tokens[i]));

    frame.delta = frame.ids.map((id, i) => frame.tokens[i] - (before.get(id) ?? 0));
    frame.delta_reconstructed = true;
  },

  rebuildMetrics() {
    if (this.frame) this.metrics = new FrameMetrics(this.frame, this.settings);
  },

  togglePlay() {
    this.playing = !this.playing;
    document.getElementById('btnPlay').textContent = this.playing ? '❚❚' : '▶';
  },

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
  STAT_GROUPS: [
    { key: 'general', label: 'General', open: true, keys: [
      'nodes', 'edges', 'tokens', 'meanTokens', 'medianTokens', 'maxTokens', 'minTokens',
      'gini', 'topDecileShare', 'tokenEntropy', 'tokenEvenness',
      'maxTokenAdded', 'maxTokenLost', 'gainers', 'losers',
      'starved', 'orphaned', 'redistributed',
      'distinctBrains', 'brainDiversity', 'distinctLineages'
    ] },
    { key: 'reproduction', label: 'Reproduction', open: true, keys: [
      'births', 'reproTokenShare', 'meanInvestedShare', 'meanChildLinks', 'handovers'
    ] },
    { key: 'blotto', label: 'Game (Blotto)', open: true, keys: [
      'totalFlow', 'meanEdgeFlow', 'maxEdgeFlow', 'selfAllocationShare',
      'revoltShare', 'spreadShare', 'revolutions', 'heldHomeShare', 'prunedEdges'
    ] },
    { key: 'structure', label: 'Structure', open: false, keys: [
      'density', 'meanDegree', 'medianDegree', 'maxDegree', 'minDegree', 'leaves',
      'cycleRank', 'loopDensity', 'bridges', 'triangles', 'transitivity',
      'dimension', 'degreeEntropy', 'degreeEvenness', 'components'
    ] }
  ],

  updateStats() {
    const container = document.getElementById('statsStrip');
    if (!this.metrics) { container.innerHTML = ''; return; }
    const s = this.metrics.summary();

    // Node counts are also given as a share of the population that entered the
    // phase — "40 births" reads very differently at 100 agents than at 4,000.
    const base = s.nodesBefore || s.nodes || 0;
    const withShare = v => (base && v !== null && v !== undefined)
      ? `${formatNumber(v)} <i>${((v / base) * 100).toFixed(1)}%</i>` : formatNumber(v);

    const int = v => formatNumber(Math.round(v));
    const pct = v => `${(v * 100).toFixed(1)}%`;
    const dec = (v, n = 2) => v.toFixed(n);

    // label and formatted value for every statistic that has one this frame
    const cells = {
      nodes: ['Nodes', formatNumber(s.nodes)],
      edges: ['Edges', formatNumber(s.edges)],
      tokens: ['Tokens', formatNumber(s.tokens)],
      meanTokens: ['Mean tokens', int(s.meanTokens)],
      medianTokens: ['Median tokens', int(s.medianTokens)],
      maxTokens: ['Richest', formatNumber(s.maxTokens)],
      minTokens: ['Poorest', formatNumber(s.minTokens)],
      gini: ['Gini', dec(s.gini, 3)],
      topDecileShare: ['Top 10% hold', pct(s.topDecileShare)],
      tokenEntropy: ['Token entropy', `${dec(s.tokenEntropy)} bits`],
      tokenEvenness: ['Token evenness', pct(s.tokenEvenness)],
      maxTokenAdded: ['Max token added', `+${formatNumber(s.maxTokenAdded)}`],
      maxTokenLost: ['Max token lost', `-${formatNumber(s.maxTokenLost)}`],
      gainers: ['Gained', withShare(s.gainers)],
      losers: ['Lost', withShare(s.losers)],
      distinctBrains: ['Distinct brains', formatNumber(s.distinctBrains)],
      brainDiversity: ['Brain diversity', pct(s.brainDiversity)],
      distinctLineages: ['Distinct lineages', formatNumber(s.distinctLineages)],

      density: ['Density', `${(s.density * 100).toFixed(2)}%`],
      meanDegree: ['Mean degree', dec(s.meanDegree)],
      medianDegree: ['Median degree', dec(s.medianDegree, 1)],
      maxDegree: ['Max degree', formatNumber(s.maxDegree)],
      minDegree: ['Min degree', formatNumber(s.minDegree)],
      leaves: ['Leaves', withShare(s.leaves)],
      cycleRank: ['Loops', formatNumber(s.cycleRank)],
      loopDensity: ['Loop density', pct(s.loopDensity)],
      bridges: ['Bridges', formatNumber(s.bridges)],
      triangles: ['Triangles', formatNumber(s.triangles)],
      transitivity: ['Clustering', dec(s.transitivity, 3)],
      dimension: ['Dimension', s.dimension === null ? '—' : dec(s.dimension)],
      degreeEntropy: ['Degree entropy', `${dec(s.degreeEntropy)} bits`],
      degreeEvenness: ['Degree evenness', pct(s.degreeEvenness)],
      components: ['Components', formatNumber(s.components)]
    };

    // Present only when the phase produced them.
    if (s.births !== null) {
      cells.births = ['Births', withShare(s.births)];
      cells.reproTokenShare = ['Tokens to offspring', pct(s.reproTokenShare)];
      cells.meanInvestedShare = ['Mean investment', pct(s.meanInvestedShare)];
      cells.meanChildLinks = ['Links per child', dec(s.meanChildLinks)];
      if (s.handovers !== null) cells.handovers = ['Handovers', formatNumber(s.handovers)];
    }
    if (s.totalFlow !== null) {
      cells.totalFlow = ['Tokens moved', formatNumber(s.totalFlow)];
      cells.meanEdgeFlow = ['Mean edge flow', dec(s.meanEdgeFlow, 1)];
      cells.maxEdgeFlow = ['Max edge flow', formatNumber(s.maxEdgeFlow)];
      cells.selfAllocationShare = ['Kept at home', pct(s.selfAllocationShare)];
      cells.revoltShare = ['Revolt tokens', pct(s.revoltShare)];
      cells.spreadShare = ['Spread doctrine', pct(s.spreadShare)];
    }
    if (s.revolutions !== null) cells.revolutions = ['Revolutions', withShare(s.revolutions)];
    if (s.heldHomeShare !== null) cells.heldHomeShare = ['Held own node', pct(s.heldHomeShare)];
    if (s.prunedEdges !== null) cells.prunedEdges = ['Pruned edges', formatNumber(s.prunedEdges)];
    if (s.starved !== null) cells.starved = ['Starved', withShare(s.starved)];
    if (s.orphaned !== null) cells.orphaned = ['Culled', withShare(s.orphaned)];
    if (s.redistributed !== null) cells.redistributed = ['Redistributed', formatNumber(s.redistributed)];

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
  },

  updateCharts() {
    if (!this.metrics) return;
    const s = this.settings;

    drawHistogram(document.getElementById('degreeHist'), Array.from(this.metrics.degree), {
      bins: 30, colormap: s.nodeColormap, reverse: s.nodeColorReverse,
      logScale: s.histDegreeX === 'log', logCount: s.histDegreeY === 'log'
    });
    drawHistogram(document.getElementById('tokenHist'), this.frame.tokens, {
      bins: 30, colormap: s.nodeColormap, reverse: s.nodeColorReverse,
      logScale: s.histTokenX === 'log', logCount: s.histTokenY === 'log'
    });
  },

  showHover(i, x, y) {
    if (i < 0) { this.hoverCard.classList.add('hidden'); return; }
    const d = this.metrics.nodeDetail(i);

    const rows = [
      `<b>Node ${d.id}</b> <span class="hint">#${d.rank} by wealth</span>`,
      `Tokens ${formatNumber(d.tokens)} <span class="hint">(${(d.tokenShare * 100).toFixed(2)}% of world)</span>`,
      `Degree ${d.degree}`,
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
  },

  applySettings(preset) {
    if (!preset) return;
    Object.assign(this.settings, preset);

    this.syncControlsFromSettings();
    this.syncAxisToggles();
    this.applyLayoutSettings();
    if (preset.dimensions) this.setDimensions(preset.dimensions);
    this.setAutoFit(this.settings.autoFit);
    this.layout.reheat(0.6);

    this.rebuildMetrics();
    this.updateCharts();
  },

  /** Push settings back into the form controls after a preset. */
  syncControlsFromSettings() {
    for (const [key, value] of Object.entries(this.settings)) {
      const el = document.getElementById(key);
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = Boolean(value);
      else el.value = value;
    }
  },

  // ------------------------------------------------------------------
  // Animation loop
  // ------------------------------------------------------------------

  resize() {
    this.renderer.resize();
    if (!this._framed && this.renderer.cssWidth > 0 && this.frame) {
      this._framed = true;
      this.renderer.fitToContent(this.layout, undefined, true);
    }
    this.redraw();
  },

  /** Paint one frame immediately, without waiting for the animation loop. */
  redraw() {
    if (this.layout && !this.layout.positionsMatchFrame) return;
    if (this.renderer.cssWidth > 0) {
      this.renderer.draw(this.frame, this.metrics, this.layout, this.settings);
    }
  },

  animate(time) {
    const dt = Math.min(0.1, (time - this.lastTime) / 1000) || 0;
    this.lastTime = time;

    // With a worker this returns immediately: the layout is advancing on its
    // own thread and the loop's only job is to draw what has arrived. Without
    // one it advances the layout here, as before.
    this.layout.tick();

    // A frame change reaches the worker before its coordinates come back. Until
    // they do, the positions we hold are ordered by the previous frame's ids,
    // and drawing the new ids against them paints one frame of nonsense. The
    // canvas simply keeps what it already shows for that moment.
    if (!this.layout.positionsMatchFrame) {
      requestAnimationFrame(t => this.animate(t));
      return;
    }

    if (this.settings.autoFit) this.renderer.fitToContent(this.layout);
    this.renderer.stepCamera();

    if (this.playing && this.visible.length) {
      const fps = Number(document.getElementById('playSpeed').value) || 6;
      this.playAccumulator += dt;
      if (this.playAccumulator >= 1 / fps) {
        this.playAccumulator = 0;
        if (this.position >= this.visible.length - 1) this.togglePlay();
        else this.goToPosition(this.position + 1);
      }
    }

    this.renderer.draw(this.frame, this.metrics, this.layout, this.settings);
    requestAnimationFrame(t => this.animate(t));
  }
};
