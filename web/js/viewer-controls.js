/*
 * The settings panel, and keeping it in step with the viewer.
 *
 * A different kind of work from the core: this file wires inputs to settings
 * and settings back to inputs, while viewer.js is about frames and drawing.
 * Nothing here decides anything — it reads a control, writes a setting, and
 * asks the viewer to catch up.
 *
 * Attached to the same Viewer rather than made an object of its own, because
 * these are called as `this.rebuildMetrics()` from every direction; separating
 * them would mean threading a reference through all of it for no gain.
 */

Object.assign(Viewer, {

  // ------------------------------------------------------------------
  // Wiring
  // ------------------------------------------------------------------

  bindControls() {
    // A control's id is the setting it drives, and what it holds follows from
    // what kind of control it is: a checkbox a boolean, a slider a number,
    // anything else text. Every one of thirty-nine calls used to pass its id
    // twice and a transform that said the same thing its type already did.
    const bind = (id, rebuild) => {
      const el = document.getElementById(id);
      if (!el) return;
      const apply = () => {
        this.settings[id] = el.type === 'checkbox' ? el.checked
          : el.type === 'range' ? Number(el.value) : el.value;
        if (rebuild) this.rebuildMetrics();
        this.updateCharts();
      };
      // A menu redraws as you move through it rather than when you commit, so
      // picking a metric is browsing rather than guessing.
      if (el.tagName === 'SELECT') {
        Metrics.onPick(el, apply);
        // A popped-open menu swallows the arrow keys, so every menu gets a
        // pair of buttons to step through it with.
        Metrics.addSteppers(el);
      } else {
        el.addEventListener(el.type === 'checkbox' ? 'change' : 'input', apply);
      }
    };

    // These change what is measured, so the frame's metrics are rebuilt. The
    // three chart menus are among them because a chart may name a quantity
    // measured before the phase, and it is the rebuild that goes and fetches
    // the frame such a quantity is read from.
    for (const id of ['nodeColorBy', 'nodeColorLog', 'nodeColormap', 'nodeColorReverse',
                      'nodeSizeBy', 'nodeSizeLog', 'edgeColorBy', 'edgeColorLog',
                      'edgeWidthBy', 'edgeWidthLog', 'distMetric', 'heatX', 'heatY']) {
      bind(id, true);
    }
    // These only change how it is drawn.
    for (const id of ['nodeSizeMin', 'nodeSizeMax', 'nodeAlpha', 'nodeOutline',
                      'nodeOutlineColor', 'nodeOutlineAlpha', 'nodeOutlineWidth',
                      'nodeGlow', 'nodeGlowColorBy', 'nodeGlowSize', 'nodeGlowStrength',
                      'edgeShow', 'edgeColormap', 'edgeColorReverse', 'edgeFlatColor',
                      'edgeWidthMin', 'edgeWidthMax', 'edgeAlpha', 'trajX', 'trajY',
                      'bgStyle', 'bgColorA', 'bgColorB', 'showLegend', 'showEdgeLegend',
                      'layoutCarry']) {
      bind(id, false);
    }

    // Layout sliders are ordinary settings; layout.applySettings pushes them
    // into the simulation so presets and the controls stay in step.
    for (const id of ['forceCharge', 'forceLink', 'forceCenter', 'forceAngular',
                      'forceDamping', 'forceTheta']) {
      const el = document.getElementById(id);
      if (!el) continue;
      el.addEventListener('input', () => {
        this.settings[id] = Number(el.value);
        this.layout.applySettings(this.settings);
        this.layout.reheat(0.6);
      });
    }

    document.getElementById('btnReheat').addEventListener('click', () => this.layout.reheat(1));
    document.getElementById('btnRelayout').addEventListener('click', () => this.layout.scatter());
    document.getElementById('btnFit').addEventListener('click', () => this.setAutoFit(!this.settings.autoFit));
    document.getElementById('btnAutoRotate')
      .addEventListener('click', () => this.setAutoRotate(!this.settings.autoRotate));

    const speed = document.getElementById('rotateSpeed');
    speed.addEventListener('input', () => {
      // Clamped rather than trusted: a number field will hand over whatever is
      // typed into it, including nothing at all.
      // An empty field is somebody midway through typing, not a request for
      // the slowest possible turn.
      if (speed.value.trim() === '') return;
      const asked = Number(speed.value);
      if (!Number.isFinite(asked)) return;
      this.settings.rotateSpeed = Math.max(1, Math.min(120, asked));
      // Changing the speed is a way of saying you want it turning.
      if (!this.settings.autoRotate) this.setAutoRotate(true);
    });
    document.getElementById('btnFullscreen').addEventListener('click', () => this.toggleFullscreen());
    document.getElementById('btnClearFocus').addEventListener('click', () => this.setFocus(null));

    document.getElementById('btnTrajLoad').addEventListener('click', () => {
      this.loadHistory([this.settings.trajX, this.settings.trajY]);
      this.updateTrajectory();
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

    // The k-core control. Its two halves change the same view, so both go
    // through one handler rather than each having its own idea of what the
    // other is set to.
    const coreOn = document.getElementById('kCoreOn');
    const coreK = document.getElementById('kCoreK');
    if (coreOn && coreK) {
      const applyCore = () => {
        const k = Math.max(1, Math.min(99, Math.round(Number(coreK.value) || 1)));
        coreK.value = k;
        this.settings.kCore = k;
        this.settings.kCoreOn = coreOn.checked && this.focusId === null;
        this.refocus(true);
      };
      coreOn.addEventListener('change', applyCore);
      coreK.addEventListener('change', () => { if (coreOn.checked) applyCore(); });
    }

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
    // menus are built from the same sections the strip under the canvas uses.
    const statOptions = RunStats.GROUPS.map(group => {
      const items = Object.entries(group.stats)
        .map(([k, label]) => `<option value="${k}">${label}</option>`)
        .join('');
      return items ? `<optgroup label="${group.label}">${items}</optgroup>` : '';
    }).join('');
    for (const id of ['trajX', 'trajY']) {
      const select = document.getElementById(id);
      if (!select) continue;
      select.innerHTML = statOptions;
      select.value = s[id];
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

  applySettings(preset) {
    if (!preset) return;
    Object.assign(this.settings, preset);

    this.syncControlsFromSettings();
    this.syncAxisToggles();
    this.layout.applySettings(this.settings);
    if (preset.dimensions) this.setDimensions(preset.dimensions);
    this.setAutoFit(this.settings.autoFit);
    // Through the setter, so the button shows what the preset asked for
    // instead of the setting and the button disagreeing.
    this.setAutoRotate(this.settings.autoRotate);
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

  /**
   * Turn the view steadily, or stop.
   *
   * Only means anything in 3D — in 2D there is no third axis to turn around,
   * so the button says so by being unavailable rather than by doing nothing
   * when pressed.
   */
  setAutoRotate(on) {
    this.settings.autoRotate = Boolean(on) && this.settings.dimensions === 3;
    const btn = document.getElementById('btnAutoRotate');
    btn.classList.toggle('active', this.settings.autoRotate);
    btn.setAttribute('aria-pressed', String(this.settings.autoRotate));

    // Whatever the camera drifts to is where it stays when this is switched
    // off, rather than springing back to where the turning began.
    if (!this.settings.autoRotate && !this.settings.autoFit) {
      this.renderer.holdCurrentView();
    }
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
  }

});
