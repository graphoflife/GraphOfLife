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
