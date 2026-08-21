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

  frameCount: 0,      // frames on disk,

  visible: [],        // frame indices passing the phase filter,

  position: 0,        // index into `visible`,

  frameIndex: 0,      // the actual frame index being shown,

  frame: null,

  metrics: null,

  cache: new Map(),

  inflight: new Map(),   // index -> in-progress fetch, so one request serves all askers,

  // Focus: the graph is cropped to what lies within a few steps of one node.
  // `fullFrame` is what was read from disk; `frame` is what is on screen, and
  // the two differ only while a focus is set.
  fullFrame: null,

  focusId: null,

  focusNeighbours: [],   // remembered while the focus node is alive, for the fallback,

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

    // Turning the view steadily, in degrees a second. A still projection of a
    // 3D graph is ambiguous — near and far look alike — and turning it is what
    // resolves the depth. Off by default: it should be asked for, not sprung
    // on someone trying to read one frame.
    autoRotate: false, rotateSpeed: 10,

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
    // The starting look. Merged in rather than applied through applySettings,
    // because that reaches for the layout and the controls, and neither exists
    // yet — the pushes at the end of this function are what deliver it.
    Object.assign(this.settings, Metrics.migrateSettings({ ...Presets.builtIn('default') }));

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

    // A flat graph has no third axis to turn around, so the button says so by
    // being unavailable rather than by doing nothing when pressed.
    const rotateBtn = document.getElementById('btnAutoRotate');
    if (rotateBtn) {
      rotateBtn.disabled = dims !== 3;
      rotateBtn.title = dims === 3
        ? 'Turn the view steadily, so a 3D graph reads as one'
        : 'Only in 3D — a flat graph has nothing to turn around';
      if (dims !== 3 && this.settings.autoRotate) this.setAutoRotate(false);
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
    // Shaken in proportion to how much of the graph is new to the layout,
    // rather than by whether carrying positions is switched on.
    //
    // The old rule read the setting and nothing else, so with carry on — the
    // default — every frame got 0.35, including the first frame of a run,
    // where nothing had been carried and every agent was starting from a
    // random point. Measured on 3,793 agents: 0.35 from cold settles at a mean
    // edge length of 0.366 of the graph's radius against 0.208 for a full
    // shake, which is the tangle that had to be reheated by hand. A full shake
    // on a single step is not the answer either — it is tidier still, but
    // moves surviving agents half a radius, and watching structure persist is
    // most of the point of carrying positions at all.
    this.layout.reheat(carry ? Math.min(1, 0.3 + 0.7 * this.layout.freshShare) : 1);

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
    // Never negative. A timestamp that goes backwards — which happens when
    // a tab is restored, and whenever the loop is driven by hand — would
    // otherwise turn the camera the wrong way and rewind playback.
    const dt = Math.max(0, Math.min(0.1, (time - this.lastTime) / 1000)) || 0;
    this.lastTime = time;

    // Nothing to do while the reader is looking at another tab.
    //
    // This loop used to run whatever was on screen. The canvas keeps its size
    // when the view is hidden, so every check that guards against drawing into
    // nothing still passed, and the Viewer went on projecting, sorting and
    // painting a few thousand agents onto a canvas nobody could see — measured
    // at sixty draws in sixty frames, 4.9ms each, while the front page was
    // trying to animate. Playback stays where it is and picks up on return.
    if (!App.isViewerActive()) {
      requestAnimationFrame(t => this.animate(t));
      return;
    }

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

    // Turned before the framing is worked out, so that with Fit view on the
    // camera is fitting the orientation about to be drawn rather than the one
    // just gone.
    if (this.settings.autoRotate && this.renderer.mode3D) {
      this.renderer.rotate((this.settings.rotateSpeed * Math.PI / 180) * dt, 0);
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
