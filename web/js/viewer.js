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
  frameCount: 0,
  frameIndex: 0,
  frame: null,
  metrics: null,
  cache: new Map(),
  playing: false,
  playAccumulator: 0,
  lastTime: 0,

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

    // The canvas has no size until the viewer tab is revealed, and a resize
    // listener alone would miss that moment. A ResizeObserver fires exactly
    // when the element gains or changes size, whatever caused it.
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
    // Every control whose id matches a settings key updates it directly.
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
    const force = (id, prop, scale = 1) => {
      const el = document.getElementById(id);
      if (!el) return;
      el.addEventListener('input', () => {
        this.layout[prop] = Number(el.value) * scale;
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
      btn.addEventListener('click', () => this.applyPreset(btn.dataset.preset));
    }
  },

  bindPlayback() {
    document.getElementById('btnFirst').addEventListener('click', () => this.goTo(0));
    document.getElementById('btnPrev').addEventListener('click', () => this.goTo(this.frameIndex - 1));
    document.getElementById('btnNext').addEventListener('click', () => this.goTo(this.frameIndex + 1));
    document.getElementById('btnLast').addEventListener('click', () => this.goTo(this.frameCount - 1));
    document.getElementById('btnPlay').addEventListener('click', () => this.togglePlay());

    const slider = document.getElementById('frameSlider');
    slider.addEventListener('input', () => this.goTo(Number(slider.value)));

    // Arrow keys drive the same navigation as the buttons.
    document.addEventListener('keydown', e => {
      if (!App.isViewerActive()) return;
      const tag = document.activeElement && document.activeElement.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;

      switch (e.key) {
        case 'ArrowLeft':  this.goTo(this.frameIndex - (e.shiftKey ? 10 : 1)); break;
        case 'ArrowRight': this.goTo(this.frameIndex + (e.shiftKey ? 10 : 1)); break;
        case 'ArrowUp':    this.goTo(this.frameIndex - 2); break;   // same phase, previous iteration
        case 'ArrowDown':  this.goTo(this.frameIndex + 2); break;
        case 'Home':       this.goTo(0); break;
        case 'End':        this.goTo(this.frameCount - 1); break;
        case ' ':          this.togglePlay(); break;
        default: return;
      }
      e.preventDefault();
    });
  },

  bindCanvas() {
    let dragging = false, lastX = 0, lastY = 0;

    this.canvas.addEventListener('mousedown', e => {
      dragging = true; lastX = e.clientX; lastY = e.clientY;
    });
    window.addEventListener('mouseup', () => { dragging = false; });
    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      this.renderer.pan(e.clientX - lastX, e.clientY - lastY);
      lastX = e.clientX; lastY = e.clientY;
    });

    this.canvas.addEventListener('wheel', e => {
      e.preventDefault();
      const rect = this.canvas.getBoundingClientRect();
      this.renderer.zoomAt(e.clientX - rect.left, e.clientY - rect.top,
                           e.deltaY < 0 ? 1.12 : 1 / 1.12);
    }, { passive: false });

    this.canvas.addEventListener('mousemove', e => {
      if (!this.frame) return;
      const rect = this.canvas.getBoundingClientRect();
      const i = this.renderer.pick(this.frame, this.layout,
                                   e.clientX - rect.left, e.clientY - rect.top);
      this.showHover(i, e.clientX - rect.left, e.clientY - rect.top);
    });
    this.canvas.addEventListener('mouseleave', () => this.hoverCard.classList.add('hidden'));
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
    this.runId = runId;
    this.cache.clear();
    this.layout.pos.clear();

    try {
      this.meta = await API.getRun(runId);
    } catch (err) {
      this.emptyEl.textContent = `Could not load run: ${err.message}`;
      this.emptyEl.style.display = '';
      return;
    }

    this.frameCount = this.meta.frame_count || 0;
    document.getElementById('activeRunLabel').textContent = this.meta.name;
    document.getElementById('runPicker').value = runId;

    if (!this.frameCount) {
      this.frame = null;
      this.emptyEl.textContent = 'This run has no recorded frames yet. Start it from the Runs tab.';
      this.emptyEl.style.display = '';
      this.updateSlider();
      return;
    }

    this.emptyEl.style.display = 'none';
    await this.goTo(Math.min(this.frameIndex, this.frameCount - 1), true);
    this.renderer.fit(this.layout.bounds());
  },

  async reload() {
    if (!this.runId) return;
    this.cache.clear();
    await this.load(this.runId);
  },

  async fetchFrame(index) {
    if (this.cache.has(index)) return this.cache.get(index);

    const frame = await API.getFrame(this.runId, index);
    this.cache.set(index, frame);

    // Bounded cache: frames carry full topology and can be large.
    if (this.cache.size > 60) {
      const oldest = this.cache.keys().next().value;
      this.cache.delete(oldest);
    }
    return frame;
  },

  async goTo(index, force = false) {
    if (!this.runId || !this.frameCount) return;
    const target = Math.max(0, Math.min(this.frameCount - 1, index));
    if (target === this.frameIndex && this.frame && !force) return;

    this.frameIndex = target;
    try {
      this.frame = await this.fetchFrame(target);
    } catch (err) {
      this.emptyEl.textContent = `Frame ${target} could not be read: ${err.message}`;
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
    slider.max = Math.max(0, this.frameCount - 1);
    slider.value = this.frameIndex;

    const label = document.getElementById('frameLabel');
    if (this.frame) {
      label.textContent =
        `Iteration ${formatNumber(this.frame.iteration)} · Phase ${this.frame.phase}` +
        ` (${this.frame.phase === 1 ? 'reproduction' : 'blotto'})` +
        ` · frame ${this.frameIndex + 1}/${this.frameCount}`;
    } else {
      label.textContent = '—';
    }
  },

  updateStats() {
    const strip = document.getElementById('statsStrip');
    if (!this.metrics) { strip.innerHTML = ''; return; }
    const s = this.metrics.summary();

    const cells = [
      ['Nodes', formatNumber(s.nodes)],
      ['Edges', formatNumber(s.edges)],
      ['Tokens', formatNumber(s.tokens)],
      ['Mean degree', s.meanDegree.toFixed(2)],
      ['Max degree', formatNumber(s.maxDegree)],
      ['Median tokens', formatNumber(Math.round(s.medianTokens))],
      ['Richest', formatNumber(s.maxTokens)],
      ['Gini', s.gini.toFixed(3)],
      ['Distinct brains', formatNumber(s.distinctBrains)],
      ['Brain diversity', `${(s.brainDiversity * 100).toFixed(1)}%`]
    ];
    if (s.births !== null) cells.push(['Births', formatNumber(s.births)]);
    if (s.revolutions !== null) cells.push(['Revolutions', formatNumber(s.revolutions)]);
    if (s.starved !== null) cells.push(['Starved', formatNumber(s.starved)]);
    if (s.orphaned !== null) cells.push(['Culled', formatNumber(s.orphaned)]);

    strip.innerHTML = cells
      .map(([k, v]) => `<div class="stat"><span class="stat-key">${k}</span><span class="stat-val">${v}</span></div>`)
      .join('');
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

  applyPreset(name) {
    const presets = {
      wealth: { nodeColorBy: 'log_tokens', nodeColormap: 'inferno', nodeSizeBy: 'tokens',
                edgeColorBy: 'constant', edgeWidthBy: 'constant', edgeAlpha: 0.18,
                bgStyle: 'radial', nodeAlpha: 0.95 },
      lineage: { nodeColorBy: 'brain_id', nodeColormap: 'turbo', nodeSizeBy: 'log_tokens',
                 edgeColorBy: 'source', edgeAlpha: 0.3, bgStyle: 'solid', nodeAlpha: 0.9 },
      structure: { nodeColorBy: 'degree', nodeColormap: 'cividis', nodeSizeBy: 'degree',
                   edgeColorBy: 'avg_degree', edgeWidthBy: 'avg_degree', edgeAlpha: 0.35,
                   bgStyle: 'linear', nodeAlpha: 0.85 },
      minimal: { nodeColorBy: 'constant', nodeColormap: 'grayscale', nodeSizeBy: 'constant',
                 edgeColorBy: 'constant', edgeWidthBy: 'constant', edgeAlpha: 0.12,
                 bgStyle: 'solid', nodeAlpha: 0.7 }
    };

    Object.assign(this.settings, presets[name] || {});
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
    // A first real measurement means the layout has never been framed; fit it
    // so opening the tab does not land on an empty-looking canvas.
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

    if (this.playing && this.frameCount) {
      const fps = Number(document.getElementById('playSpeed').value) || 6;
      this.playAccumulator += dt;
      if (this.playAccumulator >= 1 / fps) {
        this.playAccumulator = 0;
        if (this.frameIndex >= this.frameCount - 1) this.togglePlay();
        else this.goTo(this.frameIndex + 1);
      }
    }

    this.renderer.draw(this.frame, this.metrics, this.layout, this.settings);
    requestAnimationFrame(t => this.animate(t));
  }
};
