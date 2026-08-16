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
  inflight: new Map(),   // index -> in-progress fetch, so one request serves all askers
  // Focus: the graph is cropped to what lies within a few steps of one node.
  // `fullFrame` is what was read from disk; `frame` is what is on screen, and
  // the two differ only while a focus is set.
  fullFrame: null,
  focusId: null,
  focusNeighbours: [],   // remembered while the focus node is alive, for the fallback
  focusNote: '',

  playing: false,
  playAccumulator: 0,
  lastTime: 0,
  phaseFilter: 'all',

  settings: {
    nodeColorBy: 'tokens', nodeColorLog: false,
    nodeColormap: 'viridis', nodeColorReverse: false,
    nodeSizeBy: 'tokens', nodeSizeLog: false, nodeSizeMin: 1.5, nodeSizeMax: 9,
    nodeAlpha: 0.9,
    nodeOutline: false, nodeOutlineColor: '#000000',
    nodeOutlineAlpha: 0.55, nodeOutlineWidth: 0.6,
    // Glow defaults to the node's own colour rather than the outline's. The
    // effect is additive, and the default outline is black, which adds
    // nothing — switching it on would look broken rather than subtle.
    nodeGlow: false, nodeGlowColorBy: 'node',
    nodeGlowSize: 2.6, nodeGlowStrength: 0.35,
    edgeShow: true, edgeColorBy: 'constant', edgeColorLog: false,
    edgeColormap: 'cividis', edgeColorReverse: false,
    edgeFlatColor: '#5f7d95', edgeWidthBy: 'constant', edgeWidthLog: false,
    edgeWidthMin: 0.3, edgeWidthMax: 1.6, edgeAlpha: 0.25,
    bgStyle: 'solid', bgColorA: '#0d1117', bgColorB: '#1d2530',
    showLegend: true, showEdgeLegend: true, layoutCarry: true,

    // Layout lives in the settings too, so a preset restores the whole look
    // including how the graph arranges itself, not just its colours.
    forceCharge: 20, forceLink: 0.12, forceCenter: 0.012,
    forceAngular: 0.15, forceDamping: 0.86, forceTheta: 1.2,
    dimensions: 3, autoFit: true,

    // How far out from the focused node the view reaches. A setting rather
    // than plain state, so a preset carries it; which node is focused is not,
    // since node ids mean nothing across runs.
    focusRadius: 2,

    // Which quantity each chart plots, and the scale of each axis. Chart
    // metrics are domain-qualified, since `loops` means one thing for a node
    // and another for an edge.
    distMetric: 'node:tokens',
    histDistX: 'log', histDistY: 'linear',
    heatX: 'node:degree', heatY: 'node:tokens',
    histHeatX: 'linear', histHeatY: 'log', histHeatCount: 'log',

    // The trajectory plots two run statistics against each other over time.
    trajX: 'nodes', trajY: 'tokens',
    histTrajX: 'linear', histTrajY: 'linear'
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
    this.populateMetricSelects();

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
    bind('nodeColorLog', 'nodeColorLog');
    bind('nodeColormap', 'nodeColormap');
    bind('nodeColorReverse', 'nodeColorReverse');
    bind('nodeSizeBy', 'nodeSizeBy');
    bind('nodeSizeLog', 'nodeSizeLog');
    bind('nodeSizeMin', 'nodeSizeMin', num, false);
    bind('nodeSizeMax', 'nodeSizeMax', num, false);
    bind('nodeAlpha', 'nodeAlpha', num, false);
    bind('nodeOutline', 'nodeOutline', v => v, false);
    bind('nodeOutlineColor', 'nodeOutlineColor', v => v, false);
    bind('nodeOutlineAlpha', 'nodeOutlineAlpha', num, false);
    bind('nodeOutlineWidth', 'nodeOutlineWidth', num, false);
    bind('nodeGlow', 'nodeGlow', v => v, false);
    bind('nodeGlowColorBy', 'nodeGlowColorBy', v => v, false);
    bind('nodeGlowSize', 'nodeGlowSize', num, false);
    bind('nodeGlowStrength', 'nodeGlowStrength', num, false);

    bind('edgeShow', 'edgeShow', v => v, false);
    bind('edgeColorBy', 'edgeColorBy');
    bind('edgeColorLog', 'edgeColorLog');
    bind('edgeColormap', 'edgeColormap', v => v, false);
    bind('edgeColorReverse', 'edgeColorReverse', v => v, false);
    bind('edgeFlatColor', 'edgeFlatColor', v => v, false);
    bind('edgeWidthBy', 'edgeWidthBy');
    bind('edgeWidthLog', 'edgeWidthLog');

    // These rebuild rather than merely redraw, because a chart may name a
    // quantity measured before the phase, and it is the rebuild that goes and
    // fetches the frame such a quantity is read from.
    bind('distMetric', 'distMetric');
    bind('heatX', 'heatX');
    bind('heatY', 'heatY');
    bind('trajX', 'trajX', v => v, false);
    bind('trajY', 'trajY', v => v, false);
    bind('edgeWidthMin', 'edgeWidthMin', num, false);
    bind('edgeWidthMax', 'edgeWidthMax', num, false);
    bind('edgeAlpha', 'edgeAlpha', num, false);

    bind('bgStyle', 'bgStyle', v => v, false);
    bind('bgColorA', 'bgColorA', v => v, false);
    bind('bgColorB', 'bgColorB', v => v, false);
    bind('showLegend', 'showLegend', v => v, false);
    bind('showEdgeLegend', 'showEdgeLegend', v => v, false);
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
    document.getElementById('btnFullscreen').addEventListener('click', () => this.toggleFullscreen());
    document.getElementById('btnClearFocus').addEventListener('click', () => this.setFocus(null));

    document.getElementById('btnTrajLoad').addEventListener('click', async () => {
      if (this._trajectoryLoading || !this.runId) return;
      this._trajectoryLoading = true;
      this.updateTrajectory();
      try { await StatDetail.load(this.runId); }
      catch (err) { /* the chart says so on the next draw */ }
      finally { this._trajectoryLoading = false; this.updateTrajectory(); }
    });

    const radius = document.getElementById('focusRadius');
    radius.addEventListener('change', () => {
      // 99 rather than a tighter number because these graphs are stringy: on a
      // 31,567-node frame, twelve steps from a hub reached only 2,587 nodes,
      // and from a median node six steps reached 54. Their diameter measures
      // in the sixties to low hundreds, so a small ceiling cuts the view off
      // long before the neighbourhood stops being worth looking at. Nothing is
      // spent on a radius that overshoots: the search stops when it runs out
      // of new nodes, so asking for more than the graph has costs the same as
      // asking for exactly the graph.
      const value = Math.max(1, Math.min(99, Math.round(Number(radius.value) || 1)));
      radius.value = value;
      this.settings.focusRadius = value;
      if (this.focusId !== null) this.refocus(true);
    });

    // The browser owns this state — Esc and the window chrome can change it
    // without going through the button — so the button follows the event
    // rather than the other way round.
    for (const event of ['fullscreenchange', 'webkitfullscreenchange']) {
      document.addEventListener(event, () => this.syncFullscreen());
    }
    document.getElementById('btnReloadFrames').addEventListener('click', () => this.reload());

    document.getElementById('runPicker').addEventListener('change', e => {
      if (e.target.value) this.load(e.target.value);
    });

    for (const btn of document.querySelectorAll('[data-preset]')) {
      btn.addEventListener('click', () => this.applySettings(Presets.builtIn(btn.dataset.preset)));
    }
  },

  /**
   * Fill every metric menu from the shared registry.
   *
   * Doing it here rather than in the markup keeps one list of quantities: a
   * metric added to Metrics shows up in all five menus without the HTML and
   * the code drifting apart.
   */
  populateMetricSelects() {
    const s = this.settings;
    Metrics.fillSelect(document.getElementById('nodeColorBy'), 'node',
                       { extras: [Metrics.CONSTANT], selected: s.nodeColorBy });
    Metrics.fillSelect(document.getElementById('nodeSizeBy'), 'node',
                       { extras: [Metrics.CONSTANT], selected: s.nodeSizeBy });
    Metrics.fillSelect(document.getElementById('edgeColorBy'), 'edge',
                       { extras: [Metrics.CONSTANT, Metrics.INHERIT], selected: s.edgeColorBy });
    Metrics.fillSelect(document.getElementById('edgeWidthBy'), 'edge',
                       { extras: [Metrics.CONSTANT], selected: s.edgeWidthBy });

    // The trajectory reads run statistics rather than per-node metrics, so its
    // menus are built from the same groups the strip under the canvas uses.
    const statOptions = this.STAT_GROUPS.map(group => {
      const items = group.keys
        .filter(k => this.STAT_LABELS[k])
        .map(k => `<option value="${k}">${this.STAT_LABELS[k]}</option>`)
        .join('');
      return items ? `<optgroup label="${group.label}">${items}</optgroup>` : '';
    }).join('');
    for (const [id, key] of [['trajX', 'trajX'], ['trajY', 'trajY']]) {
      const select = document.getElementById(id);
      if (!select) continue;
      select.innerHTML = statOptions;
      select.value = s[key];
    }

    Metrics.fillDomainSelect(document.getElementById('distMetric'), s.distMetric);
    Metrics.fillDomainSelect(document.getElementById('heatX'), s.heatX);
    Metrics.fillDomainSelect(document.getElementById('heatY'), s.heatY);
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
        case 'f': case 'F': this.toggleFullscreen(); break;
        default: return;
      }
      e.preventDefault();
    });
  },

  // ------------------------------------------------------------------
  // Focus: one node's neighbourhood, a few steps out
  // ------------------------------------------------------------------

  /**
   * Crop a frame to what lies within `focusRadius` steps of the focused node.
   *
   * Everything on screen then describes the neighbourhood rather than the
   * world: the layout spreads only these nodes, and the statistics, charts and
   * hover card are all computed from the cropped frame. That is the point — a
   * neighbourhood you can actually read — but it does mean a node at the edge
   * of the ball shows the degree it has *here*, not the degree it has in the
   * whole graph. What you see is what is measured.
   *
   * The focused node does not always survive the phase. It is followed to a
   * neighbour when it dies, and the whole graph comes back only when the
   * neighbourhood went with it.
   */
  focusFrame(full) {
    if (!full || this.focusId === null) return full;

    const adjacency = new Map();
    for (const id of full.ids) adjacency.set(id, []);
    for (const [a, b] of full.edges) {
      if (a === b) continue;
      const la = adjacency.get(a), lb = adjacency.get(b);
      if (la) la.push(b);
      if (lb) lb.push(a);
    }

    let anchor = this.focusId;
    if (!adjacency.has(anchor)) {
      // Gone this phase. Its neighbours from the last frame it was in are the
      // only handle left on where it was, so the densest survivor inherits the
      // focus — that keeps the view in the thick of the same neighbourhood
      // rather than on whichever id happened to be listed first.
      const gone = anchor;
      let best = null, bestDegree = -1;
      for (const id of this.focusNeighbours) {
        const list = adjacency.get(id);
        if (!list) continue;
        if (list.length > bestDegree) { best = id; bestDegree = list.length; }
      }
      if (best === null) {
        this.setFocus(null, `node ${gone} and everything around it is gone \u2014 showing the whole graph`);
        return full;
      }
      anchor = best;
      this.focusId = best;
      this.focusNote = `node ${gone} died \u2014 following neighbour ${best}`;
    }

    this.focusNeighbours = adjacency.get(anchor).slice();

    // The ball, breadth first.
    const inBall = new Set([anchor]);
    let layer = [anchor];
    for (let step = 0; step < this.settings.focusRadius && layer.length; step++) {
      const next = [];
      for (const u of layer) {
        for (const v of adjacency.get(u)) {
          if (inBall.has(v)) continue;
          inBall.add(v);
          next.push(v);
        }
      }
      layer = next;
    }

    const keep = [];
    for (let i = 0; i < full.ids.length; i++) if (inBall.has(full.ids[i])) keep.push(i);
    const take = arr => (arr ? keep.map(i => arr[i]) : undefined);

    const sub = {
      iteration: full.iteration,
      phase: full.phase,
      nodes_before: full.nodes_before,
      ids: keep.map(i => full.ids[i]),
      tokens: take(full.tokens),
      brain_ids: take(full.brain_ids),
      parent_brain_ids: take(full.parent_brain_ids),
      parent_ids: take(full.parent_ids),
      edges: full.edges.filter(([a, b]) => inBall.has(a) && inBall.has(b)),
      cleanup: full.cleanup,
      previous: full.previous,
      focusAnchor: anchor
    };
    if (full.delta) sub.delta = take(full.delta);

    // Decisions are filtered to the agents on screen, so the reproduction and
    // game statistics describe this neighbourhood too rather than the world.
    const d = full.decisions;
    if (d) {
      sub.decisions = {};
      if (d.births) sub.decisions.births = d.births.filter(b => inBall.has(b.agent));
      if (d.rewires) sub.decisions.rewires = d.rewires.filter(r => inBall.has(r.agent));
      if (d.allocations) sub.decisions.allocations = d.allocations.filter(a => inBall.has(a.agent));
      if (d.winners) sub.decisions.winners = d.winners.filter(w => inBall.has(w.node));
      if (d.pruned_edges) sub.decisions.pruned_edges = d.pruned_edges;
    }

    // The frame-level rewire count is a whole-graph number; dropping it makes
    // the stat fall back to counting the records that survived the crop.
    sub.summary = {
      nodes: sub.ids.length,
      edges: sub.edges.length,
      tokens: sub.tokens.reduce((a, b) => a + b, 0)
    };
    return sub;
  },

  /** Point the view at a node, or at nothing. */
  setFocus(id, note = '') {
    this.focusId = id;
    this.focusNote = note;
    if (id === null) this.focusNeighbours = [];
    this.refocus(true);
  },

  /** Rebuild what is on screen from the frame that was read. */
  refocus(reframe = false) {
    if (!this.fullFrame) { this.updateFocusUi(); return; }

    this.frame = this.focusFrame(this.fullFrame);
    // Handed over together, so nothing is ever drawn against the other one's
    // node ordering.
    this.layout.setFrame(this.frame.ids, this.frame.edges, this.frame.parent_ids,
                         this.settings.layoutCarry);
    this.layout.reheat(reframe ? 1 : 0.5);

    this.rebuildMetrics();
    this.updateStats();
    this.updateCharts();
    this.updateFocusUi();
    if (reframe) this.setAutoFit(true);
  },

  updateFocusUi() {
    const note = document.getElementById('focusNote');
    const clear = document.getElementById('btnClearFocus');
    const focused = this.focusId !== null;

    if (clear) clear.disabled = !focused;
    if (!note) return;

    if (!focused) { note.textContent = this.focusNote || ''; return; }
    const shown = this.frame ? this.frame.ids.length : 0;
    const whole = this.fullFrame ? this.fullFrame.ids.length : 0;
    note.textContent =
      `Focus: node ${this.focusId}, ${this.settings.focusRadius} step`
      + `${this.settings.focusRadius === 1 ? '' : 's'} out \u2014 `
      + `${formatNumber(shown)} of ${formatNumber(whole)} nodes`
      + (this.focusNote ? ` \u00b7 ${this.focusNote}` : '');
  },

  // ------------------------------------------------------------------
  // Fullscreen
  // ------------------------------------------------------------------

  /** The element currently filling the screen, whatever the browser calls it. */
  get fullscreenElement() {
    return document.fullscreenElement || document.webkitFullscreenElement || null;
  },

  /**
   * Fill the screen with the viewer, or give it back.
   *
   * The whole two-pane layout goes fullscreen rather than the canvas alone, so
   * the settings and the playbar come with it — a graph you cannot recolour or
   * step through is a screenshot, not a view.
   */
  toggleFullscreen() {
    const target = document.getElementById('viewerLayout');
    if (!target) return;

    if (this.fullscreenElement) {
      const exit = document.exitFullscreen || document.webkitExitFullscreen;
      if (exit) exit.call(document);
      return;
    }

    const request = target.requestFullscreen || target.webkitRequestFullscreen;
    if (!request) {
      this.emptyEl.textContent = 'This browser will not allow fullscreen here.';
      return;
    }
    // Refused when not driven by a real click, and on some embedded views at
    // any time; there is nothing to recover, so just leave the view as it was.
    Promise.resolve(request.call(target)).catch(() => {});
  },

  /** Follow the browser's idea of fullscreen, however it was changed. */
  syncFullscreen() {
    const target = document.getElementById('viewerLayout');
    const on = this.fullscreenElement === target;

    target.classList.toggle('is-fullscreen', on);
    const button = document.getElementById('btnFullscreen');
    if (button) {
      button.setAttribute('aria-pressed', String(on));
      button.textContent = on ? 'Exit fullscreen' : 'Fullscreen';
    }

    // The canvas has just changed size by a lot. The ResizeObserver catches
    // this on its own, but not before the next frame is drawn, and refitting
    // here keeps the graph from being briefly framed for the old box.
    this.resize();
    if (this.settings.autoFit) this.renderer.fitToContent(this.layout, undefined, true);
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
    let dragging = false, rotating = false, lastX = 0, lastY = 0, moved = 0;

    this.canvas.addEventListener('mousedown', e => {
      // Alt-drag or the middle button orbits; plain drag always pans.
      rotating = this.renderer.mode3D && (e.altKey || e.button === 1);
      dragging = !rotating;
      lastX = e.clientX; lastY = e.clientY;
      moved = 0;
      if (e.button === 1) e.preventDefault();
    });

    window.addEventListener('mouseup', () => { dragging = rotating = false; });

    window.addEventListener('mousemove', e => {
      if (!dragging && !rotating) return;
      const dx = e.clientX - lastX, dy = e.clientY - lastY;
      lastX = e.clientX; lastY = e.clientY;
      moved += Math.abs(dx) + Math.abs(dy);

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

    // A click is a mousedown and mouseup in about the same place. Panning uses
    // the same button, so without the distance test every pan would land on
    // whatever node it finished over.
    this.canvas.addEventListener('click', e => {
      if (moved > 4 || !this.frame) return;
      const rect = this.canvas.getBoundingClientRect();
      const i = this.renderer.pick(this.frame, this.layout,
                                   e.clientX - rect.left, e.clientY - rect.top);
      if (i >= 0) this.setFocus(this.frame.ids[i]);
      else if (this.focusId !== null) this.setFocus(null);
    });

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
      this.inflight.clear();
      // Node ids from the previous run mean nothing here.
      this.focusId = null;
      this.focusNeighbours = [];
      this.focusNote = '';
      this.fullFrame = null;
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
    this.inflight.clear();
    StatDetail.invalidate(this.runId);
    await this.load(this.runId);
  },

  async fetchFrame(index) {
    if (this.cache.has(index)) return this.cache.get(index);
    // Sharing the promise, not just the result: scrubbing quickly asks for the
    // same frame several times before the first answer lands, and each of
    // those used to become its own request.
    const inflight = this.inflight.get(index);
    if (inflight) return inflight;

    const request = API.getFrame(this.runId, index).then(
      frame => {
        this.cache.set(index, frame);
        // Bounded cache: frames carry full topology and can be large.
        if (this.cache.size > 60) {
          this.cache.delete(this.cache.keys().next().value);
        }
        this.inflight.delete(index);
        return frame;
      },
      err => { this.inflight.delete(index); throw err; }
    );
    this.inflight.set(index, request);
    return request;
  },

  /**
   * Warm the frames the reader is about to reach.
   *
   * Reading a frame is the slowest part of moving between them — decompressing
   * and parsing twenty thousand nodes measured anywhere from 55ms to 400ms,
   * far more than everything the browser then does with it. Almost all of that
   * can happen while the current frame is still on screen, because which frame
   * comes next is not a guess: playback runs one way, and so does holding down
   * an arrow key.
   *
   * Deliberately not awaited. A prefetch that fails or arrives late costs
   * nothing, since the real fetch will simply find it missing and ask again.
   */
  prefetch(fromPosition, direction, count = 2) {
    if (!this.runId || !this.visible.length || !direction) return;
    for (let k = 1; k <= count; k++) {
      const position = fromPosition + direction * k;
      if (position < 0 || position >= this.visible.length) return;
      const index = this.visible[position];
      if (this.cache.has(index) || this.inflight.has(index)) continue;
      this.fetchFrame(index).catch(() => {});
    }
  },

  async goToPosition(position, force = false) {
    if (!this.runId || !this.visible.length) return;
    const target = Math.max(0, Math.min(this.visible.length - 1, position));
    if (target === this.position && this.frame && !force) return;

    // Which way the reader is moving, so the frames ahead can be warmed.
    const heading = Math.sign(target - this.position) || this._heading || 1;
    this._heading = heading;

    this.position = target;
    const index = this.visible[target];
    this.frameIndex = index;
    this.prefetch(target, heading);

    let frame;
    try {
      frame = await this.fetchFrame(index);
    } catch (err) {
      if (this.position !== target) return;
      this.emptyEl.textContent = `Frame ${index} could not be read: ${err.message}`;
      this.emptyEl.style.display = '';
      return;
    }

    // Scrubbing faster than frames can be read leaves several of these in
    // flight at once, and they do not finish in the order they were asked
    // for. Whichever was asked for last is the one the reader wants, so an
    // older answer arriving late is dropped rather than painted over it.
    if (this.position !== target) return;

    // Everything that might wait goes here, while the page is still drawing the
    // frame it already had. Adopting the new one first and reading afterwards
    // opened a gap: a real fetch lets an animation frame run, and the page
    // would paint the new frame's edges against the previous frame's
    // positions — every edge joining whichever nodes happened to occupy those
    // slots, which is the burst of clutter that lasted a frame.
    await this.ensureDelta(frame, index);
    if (this.needsPreviousFrame()) await this.ensurePrevious(frame, index);
    if (this.position !== target) return;

    // From here to setFrame there is no await, so what the page draws and what
    // the layout holds change together or not at all.
    this.fullFrame = frame;
    this.focusNote = '';
    this.frame = this.focusFrame(frame);
    this.emptyEl.style.display = 'none';
    const carry = this.settings.layoutCarry && !this._dropPositions;
    this._dropPositions = false;
    this.layout.setFrame(this.frame.ids, this.frame.edges, this.frame.parent_ids, carry);
    this.layout.reheat(this.settings.layoutCarry ? 0.35 : 1);

    this.rebuildMetrics();
    this.updateSlider();
    this.updateStats();
    this.updateCharts();
    this.updateFocusUi();
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
    if (!this.frame) return;
    this.metrics = new FrameMetrics(this.frame, this.settings);

    // A "before the phase" metric reads the frame that came before this one.
    // Fetch it once, then rebuild on top of it — the view stays usable in the
    // meantime, showing the metric as absent rather than blocking on a read.
    if (this.needsPreviousFrame() && this.frame.previous === undefined) {
      const frame = this.frame;
      this.ensurePrevious(frame, this.frameIndex).then(() => {
        if (this.frame !== frame) return;    // the reader has moved on
        this.metrics = new FrameMetrics(frame, this.settings);
        this.updateCharts();
        this.updateStats();
      });
    }
  },

  /**
   * Whether the positions on hand really describe the frame about to be drawn.
   *
   * Two things have to hold. The layout's coordinates must belong to the frame
   * the layout was last given — that is the generation check — and that frame
   * must be the one the page is showing. The second half is the one that bit:
   * the layout can be perfectly self-consistent on a frame the viewer has
   * already moved past, and drawing then indexes one frame's edges into
   * another frame's coordinates.
   */
  get readyToDraw() {
    if (!this.layout.positionsMatchFrame) return false;
    if (!this.frame) return true;
    return this.layout.ids === this.frame.ids;
  },

  /** Whether anything currently on screen is measured before the phase. */
  needsPreviousFrame() {
    const s = this.settings;
    const keys = [
      s.nodeColorBy, s.nodeSizeBy,
      Metrics.parse(s.distMetric).key,
      Metrics.parse(s.heatX).key,
      Metrics.parse(s.heatY).key
    ];
    return keys.some(k => Metrics.needsPrevious(k));
  },

  /**
   * Attach the frame this one followed, which is the state the phase started
   * from.
   *
   * Set to null rather than left undefined when there is nothing usable, so
   * the lookup is not retried on every rebuild. With export_every above one
   * the previous recorded frame is several iterations back rather than the
   * state this phase began in, and comparing against it would be wrong rather
   * than merely approximate, so that case is refused outright.
   */
  async ensurePrevious(frame, index) {
    if (!frame || frame.previous !== undefined) return;
    if (index <= 0) { frame.previous = null; return; }
    if (this.meta && this.meta.config && (this.meta.config.export_every || 1) !== 1) {
      frame.previous = null;
      return;
    }
    try {
      frame.previous = await this.fetchFrame(index - 1);
    } catch (err) {
      frame.previous = null;
    }
  },

  togglePlay() {
    // Whatever was in flight no longer paces anything.
    if (this.playing) this._stepping = false;
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
    distinctLineages: 'Distinct lineages',
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
    births: 'Births',
    reproTokenShare: 'Tokens to offspring',
    meanInvestedShare: 'Mean investment',
    meanChildLinks: 'Links per child',
    handovers: 'Handovers',
    rewires: 'Rewires',
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
      'distinctBrains', 'brainDiversity', 'distinctLineages'
    ] },
    { key: 'reproduction', label: 'Reproduction', open: true, keys: [
      'births', 'reproTokenShare', 'meanInvestedShare', 'meanChildLinks',
      'handovers', 'rewires'
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
    ] }
  ],

  updateStats() {
    const container = document.getElementById('statsStrip');
    if (!this.metrics) { container.innerHTML = ''; return; }

    // Whether the reader currently has the Structure group open decides
    // whether its statistics are worth computing at all: they cost more than
    // everything else on this strip put together.
    const existing = container.querySelector('.stat-group[data-group="structure"]');
    const structureGroup = this.STAT_GROUPS.find(g => g.key === 'structure');
    const structureOpen = existing ? existing.open : Boolean(structureGroup && structureGroup.open);
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
      distinctLineages: [this.STAT_LABELS.distinctLineages, formatNumber(s.distinctLineages)],

      density: [this.STAT_LABELS.density, `${(s.density * 100).toFixed(2)}%`],
      meanDegree: [this.STAT_LABELS.meanDegree, dec(s.meanDegree)],
      medianDegree: [this.STAT_LABELS.medianDegree, dec(s.medianDegree, 1)],
      maxDegree: [this.STAT_LABELS.maxDegree, formatNumber(s.maxDegree)],
      minDegree: [this.STAT_LABELS.minDegree, formatNumber(s.minDegree)],
      leaves: [this.STAT_LABELS.leaves, withShare(s.leaves)],
      degreeEntropy: [this.STAT_LABELS.degreeEntropy, `${dec(s.degreeEntropy)} bits`],
      degreeEvenness: [this.STAT_LABELS.degreeEvenness, pct(s.degreeEvenness)]
    };

    // Only computed while the Structure group is open, since walking the whole
    // graph costs more than the rest of this strip together.
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
    if (s.rewires !== null) cells.rewires = [this.STAT_LABELS.rewires, formatNumber(s.rewires)];
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
    const group = container.querySelector('.stat-group[data-group="structure"]');
    if (group && !structureOpen) {
      group.addEventListener('toggle', () => { if (group.open) this.updateStats(); }, { once: true });
    }
  },

  /**
   * Pair two run statistics into a path through time.
   *
   * The two are not always recorded together. Most are written on both phases,
   * some only on the reproduction phase — births, what was invested, rewires —
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
      if (d.rewire) {
        rows.push(`Gave its edge to ${d.rewire.edge} over to ${d.rewire.to}`);
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
    Object.assign(this.settings, Metrics.migrateSettings({ ...preset }));

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
    if (!this.readyToDraw) {
      requestAnimationFrame(t => this.animate(t));
      return;
    }

    if (this.settings.autoFit) this.renderer.fitToContent(this.layout);
    this.renderer.stepCamera();

    if (this.playing && this.visible.length) {
      const fps = Number(document.getElementById('playSpeed').value) || 6;
      this.playAccumulator += dt;

      if (this.playAccumulator >= 1 / fps) {
        if (this._stepping) {
          // The previous frame has not arrived yet. Asking for the next one
          // anyway is worse than useless: goToPosition moves the position at
          // once but adopts the frame only when it lands, and a request whose
          // position has already moved on is dropped as stale. Firing faster
          // than frames can be read therefore superseded every one of them,
          // and the view stopped while the requests carried on — at thirty a
          // second, sixty-three asked for and fourteen ever shown.
          //
          // Held at the threshold rather than left to accumulate, so playback
          // picks straight up when the frame lands instead of lurching through
          // the backlog it built while waiting.
          this.playAccumulator = 1 / fps;
        } else if (this.position >= this.visible.length - 1) {
          this.playAccumulator = 0;
          this.togglePlay();
        } else {
          this.playAccumulator = 0;
          this._stepping = true;
          this.goToPosition(this.position + 1)
            .finally(() => { this._stepping = false; });
        }
      }
    }

    this.renderer.draw(this.frame, this.metrics, this.layout, this.settings);
    requestAnimationFrame(t => this.animate(t));
  }
};
