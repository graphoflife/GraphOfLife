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
    showLegend: true, layoutCarry: true
  },

  init() {
    this.canvas = document.getElementById('graphCanvas');
    this.renderer = new GraphRenderer(this.canvas);
    this.layout = new ForceLayout();
    this.emptyEl = document.getElementById('canvasEmpty');
    this.hoverCard = document.getElementById('hoverCard');

    populateColormapSelect(document.getElementById('nodeColormap'), this.settings.nodeColormap);
    populateColormapSelect(document.getElementById('edgeColormap'), this.settings.edgeColormap);

    this.bindControls();
    this.bindPlayback();
    this.bindCanvas();
    this.bindToggles();
    this.bindPresets();
    this.refreshPresetList();

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

    // Layout forces feed the simulation rather than the settings object.
    const force = (id, prop) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('input', () => {
        this.layout[prop] = Number(el.value);
        this.layout.reheat(0.6);
      });
    };
    force('forceCharge', 'charge');
    force('forceLink', 'linkStrength');
    force('forceCenter', 'centerStrength');
    force('forceAngular', 'angularStrength');
    force('forceDamping', 'damping');

    document.getElementById('btnReheat').addEventListener('click', () => this.layout.reheat(1));
    document.getElementById('btnRelayout').addEventListener('click', () => this.layout.scatter());
    document.getElementById('btnFit').addEventListener('click', () => this.renderer.fit(this.layout.bounds()));
    document.getElementById('btnReloadFrames').addEventListener('click', () => this.reload());

    document.getElementById('runPicker').addEventListener('change', e => {
      if (e.target.value) this.load(e.target.value);
    });

    for (const btn of document.querySelectorAll('[data-preset]')) {
      btn.addEventListener('click', () => this.applySettings(Presets.BUILT_IN[btn.dataset.preset]));
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

      if (rotating) this.renderer.rotate(dx * 0.007, dy * 0.007);
      else this.renderer.pan(dx, dy);
    });

    // Middle-click drag would otherwise trigger autoscroll.
    this.canvas.addEventListener('auxclick', e => { if (e.button === 1) e.preventDefault(); });

    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      this.renderer.zoomAt(e.clientX - rect.left, e.clientY - rect.top,
                           e.deltaY < 0 ? 1.12 : 1 / 1.12);
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
    this.layout.setDimensions(dims);
    this.renderer.setMode3D(dims === 3);

    for (const btn of document.querySelectorAll('#dimToggle .seg-btn')) {
      btn.classList.toggle('active', Number(btn.dataset.dim) === dims);
    }
    document.getElementById('dimHint').textContent = dims === 3
      ? 'Drag to pan · alt-drag or middle-drag to orbit · scroll to zoom'
      : 'Drag to pan · scroll to zoom';

    this.renderer.fit(this.layout.bounds());
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
      this.layout.pos.clear();
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
    this.renderer.fit(this.layout.bounds());
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

    this.emptyEl.style.display = 'none';
    this.layout.setFrame(this.frame.ids, this.frame.edges,
                         this.frame.parent_ids, this.settings.layoutCarry);
    this.layout.reheat(this.settings.layoutCarry ? 0.35 : 1);

    this.rebuildMetrics();
    this.updateSlider();
    this.updateStats();
    this.updateCharts();
    if (!StatDetail.el.classList.contains('hidden')) StatDetail.redraw();
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

  updateStats() {
    const strip = document.getElementById('statsStrip');
    if (!this.metrics) { strip.innerHTML = ''; return; }
    const s = this.metrics.summary();

    // Node counts are also given as a share of the population that entered the
    // phase — "40 births" reads very differently at 100 agents than at 4,000.
    const base = s.nodesBefore || s.nodes || 0;
    const share = v => (base && v !== null) ? ` (${((v / base) * 100).toFixed(1)}%)` : '';

    const cells = [
      ['nodes', 'Nodes', formatNumber(s.nodes)],
      ['edges', 'Edges', formatNumber(s.edges)],
      ['tokens', 'Tokens', formatNumber(s.tokens)],
      ['meanDegree', 'Mean degree', s.meanDegree.toFixed(2)],
      ['maxDegree', 'Max degree', formatNumber(s.maxDegree)],
      ['medianTokens', 'Median tokens', formatNumber(Math.round(s.medianTokens))],
      ['maxTokens', 'Richest', formatNumber(s.maxTokens)],
      ['gini', 'Gini', s.gini.toFixed(3)],
      ['distinctBrains', 'Distinct brains', formatNumber(s.distinctBrains)],
      ['brainDiversity', 'Brain diversity', `${(s.brainDiversity * 100).toFixed(1)}%`]
    ];
    if (s.births !== null) cells.push(['births', 'Births', formatNumber(s.births) + share(s.births)]);
    if (s.revolutions !== null) cells.push(['revolutions', 'Revolutions', formatNumber(s.revolutions) + share(s.revolutions)]);
    if (s.starved !== null) cells.push(['starved', 'Starved', formatNumber(s.starved) + share(s.starved)]);
    if (s.orphaned !== null) cells.push(['orphaned', 'Culled', formatNumber(s.orphaned) + share(s.orphaned)]);

    strip.innerHTML = cells.map(([key, label, value]) =>
      `<button class="stat" data-stat="${key}" data-label="${label}" title="Click for an explanation and its history">
         <span class="stat-key">${label}</span><span class="stat-val">${value}</span>
       </button>`).join('');

    for (const el of strip.querySelectorAll('.stat')) {
      el.addEventListener('click', () => StatDetail.open(el.dataset.stat, el.dataset.label));
    }
  },

  updateCharts() {
    if (!this.metrics) return;
    drawHistogram(document.getElementById('degreeHist'), Array.from(this.metrics.degree), {
      bins: 30, colormap: this.settings.nodeColormap, reverse: this.settings.nodeColorReverse
    });
    drawHistogram(document.getElementById('tokenHist'), this.frame.tokens, {
      bins: 30, colormap: this.settings.nodeColormap,
      reverse: this.settings.nodeColorReverse, logScale: true
    });
  },

  showHover(i, x, y) {
    if (i < 0) { this.hoverCard.classList.add('hidden'); return; }
    const f = this.frame;

    this.hoverCard.innerHTML = `
      <b>Node ${f.ids[i]}</b><br>
      Tokens ${formatNumber(f.tokens[i])}<br>
      Degree ${this.metrics.degree[i]}<br>
      Brain ${f.brain_ids[i]} <span class="hint">(from ${f.parent_brain_ids[i]})</span><br>
      Spawned by ${f.parent_ids[i] >= 0 ? f.parent_ids[i] : '—'}
    `;
    this.hoverCard.style.left = `${x + 14}px`;
    this.hoverCard.style.top = `${y + 14}px`;
    this.hoverCard.classList.remove('hidden');
  },

  applySettings(preset) {
    if (!preset) return;
    Object.assign(this.settings, preset);
    this.syncControlsFromSettings();
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
      this.renderer.fit(this.layout.bounds());
    }
    this.redraw();
  },

  /** Paint one frame immediately, without waiting for the animation loop. */
  redraw() {
    if (this.renderer.cssWidth > 0) {
      this.renderer.draw(this.frame, this.metrics, this.layout, this.settings);
    }
  },

  animate(time) {
    const dt = Math.min(0.1, (time - this.lastTime) / 1000) || 0;
    this.lastTime = time;

    this.layout.tick();

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
