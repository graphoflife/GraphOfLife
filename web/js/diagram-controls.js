/*
 * The Diagrams tab's control bar: every input that decides what a chart shows.
 *
 * Split out of diagrams.js, which had grown past a thousand lines, along the
 * seam it already had — nothing here touches a canvas, and nothing in
 * diagrams.js builds an element. What is left there is the tab's state, the
 * data it pulls, and the drawing; what is here is the DOM that drives it.
 *
 * These are methods of Diagrams rather than a separate thing with its own
 * state: they read and write that object's settings directly and call back
 * into it to redraw. Mixed in at the bottom of this file, so `this` means what
 * it means everywhere else in that object and the split costs the reader
 * nothing but a second file to open.
 */
const DiagramControls = {

  /**
   * Build this tab's controls.
   *
   * Rebuilt on every tab change rather than kept as four hidden panels: the
   * tabs share almost no controls, and four sets of stale inputs is four sets
   * of state to keep in step with the settings they are meant to reflect.
   */
  controls() {
    const bar = this.controlsEl;
    bar.replaceChildren();
    const s = this.now;

    const field = (label, node) => {
      const wrap = document.createElement('label');
      wrap.className = 'lineage-filter';
      if (label) wrap.append(document.createTextNode(label));
      wrap.append(node);
      bar.append(wrap);
      return node;
    };
    const select = (options, value, onChange) => {
      const el = document.createElement('select');
      for (const [v, text] of options) {
        const option = document.createElement('option');
        option.value = v; option.textContent = text;
        el.append(option);
      }
      el.value = value;
      Metrics.onPick(el, onChange);
      return el;
    };
    const number = (value, min, onChange) => {
      const el = document.createElement('input');
      el.type = 'number'; el.min = String(min); el.value = value === null ? '' : String(value);
      el.addEventListener('change', () => onChange(el.value === '' ? null : Number(el.value)));
      return el;
    };
    const toggle = (label, on, onChange) => {
      const el = document.createElement('button');
      el.type = 'button';
      el.className = 'axis-btn' + (on ? ' active' : '');
      el.textContent = label;
      el.addEventListener('click', () => onChange(!on));
      return el;
    };

    if (this.tab.kind === 'frame') {
      const metricSelect = () => {
        const el = document.createElement('select');
        Metrics.fillDomainSelect(el, null);
        return el;
      };
      if (this.active === 'histogram') {
        const el = field('Metric', metricSelect());
        el.value = s.metric;
        Metrics.onPick(el, v => { s.metric = v; this.draw(); });
        Metrics.addSteppers(el);
      } else {
        const x = field('x', metricSelect());
        x.value = s.x;
        Metrics.onPick(x, v => { s.x = v; this.draw(); });
        Metrics.addSteppers(x);
        const y = field('y', metricSelect());
        y.value = s.y;
        Metrics.onPick(y, v => { s.y = v; this.draw(); });
        Metrics.addSteppers(y);
      }

      // Which moment. Blank means the last recorded one, which is the useful
      // default — you almost always want to see where a run got to.
      field('At iteration', number(s.iteration, 0, v => {
        s.iteration = v; this.refresh();
      })).placeholder = 'last';
      field('summed over', number(s.span, 1, v => {
        s.span = Math.max(1, v || 1); this.refresh();
      })).title = 'How many consecutive iterations to pool into one distribution';
      bar.lastChild.append(document.createTextNode(' iterations'));

      const axes = document.createElement('span');
      axes.className = 'axis-toggles';
      axes.append(toggle('log x', s.logX, v => { s.logX = v; this.controls(); this.draw(); }));
      axes.append(toggle('log y', s.logY, v => { s.logY = v; this.controls(); this.draw(); }));
      if (this.active === 'heatmap') {
        axes.append(toggle('log n', s.logCount,
                           v => { s.logCount = v; this.controls(); this.draw(); }));
      }
      bar.append(axes);
    }

    if (this.active === 'correlate') {
      const keys = this.statOptions();
      const x = field('x', select(keys, s.x, v => { s.x = v; this.draw(); }));
      const y = field('y', select(keys, s.y, v => { s.y = v; this.draw(); }));
      x.title = 'Horizontal axis'; y.title = 'Vertical axis';
      field('Phase', select([['all', 'Both phases'], ['1', 'Reproduction'], ['2', 'Game']],
                            s.phase, v => { s.phase = v; this.draw(); }));
      const axes = document.createElement('span');
      axes.className = 'axis-toggles';
      axes.append(toggle('log x', s.logX, v => { s.logX = v; this.controls(); this.draw(); }));
      axes.append(toggle('log y', s.logY, v => { s.logY = v; this.controls(); this.draw(); }));
      bar.append(axes);
    }

    if (this.active === 'timeline') {
      const add = document.createElement('button');
      add.type = 'button';
      add.className = 'ghost small';
      add.textContent = 'Add line';
      add.addEventListener('click', () => {
        const previous = s.lines[s.lines.length - 1];
        s.lines.push(previous
          ? { ...previous }
          : { run: this.runId || (this.runs[0] && this.runs[0].id) || '',
              stat: 'nodes', phase: 'all', stretch: false, maxFrames: null });
        this.controls();
        this.refresh();
      });
      bar.append(add);

      field('from iteration', number(s.cutoff, 0, v => {
        s.cutoff = Math.max(0, v || 0); this.controls(); this.draw();
      })).title = 'Drop the opening iterations. A run begins by shaking out the '
                + 'graph it was seeded with, and that first swing is larger than '
                + 'anything after it, so on a shared scale it flattens the rest.';

      const axes = document.createElement('span');
      axes.className = 'axis-toggles';
      axes.append(toggle('log x', s.logX, v => { s.logX = v; this.controls(); this.draw(); }));
      axes.append(toggle('log y', s.logY, v => { s.logY = v; this.controls(); this.draw(); }));
      bar.append(axes);
    }

    // Shared by every tab: what it is called, what it is compared against,
    // how it is coloured, and getting a picture out.
    const name = document.createElement('input');
    name.type = 'text';
    name.className = 'diagram-title';
    name.value = s.title;
    name.placeholder = this.defaultTitle();
    name.title = 'Chart title. Left blank it describes itself.';
    name.addEventListener('change', () => { s.title = name.value.trim(); this.draw(); });
    field('Title', name);

    const guideAxis = document.createElement('select');
    for (const [v, text] of [['x', 'x'], ['y', 'y']]) {
      const option = document.createElement('option');
      option.value = v; option.textContent = text;
      guideAxis.append(option);
    }
    const guideAt = document.createElement('input');
    guideAt.type = 'number';
    guideAt.step = 'any';
    // Its own class, not the per-line cap's: they are different fields that
              // happen to be the same size, and sharing a name makes each one
              // findable only by counting.
    guideAt.className = 'diagram-guide';
    guideAt.placeholder = 'value';
    const guideAdd = document.createElement('button');
    guideAdd.type = 'button';
    guideAdd.className = 'ghost small';
    guideAdd.textContent = 'Add line at';
    guideAdd.title = 'A constant line to compare against — a threshold, a target';
    guideAdd.addEventListener('click', () => {
      const at = Number(guideAt.value);
      if (!Number.isFinite(at) || guideAt.value === '') return;
      s.guides.push({ axis: guideAxis.value, at });
      guideAt.value = '';
      this.controls();
      this.draw();
    });
    bar.append(guideAdd);
    field('', guideAxis);
    field('=', guideAt);

    const gridToggle = toggle('grid', s.grid, v => {
      s.grid = v; this.controls(); this.draw();
    });

    const style = document.createElement('span');
    style.append(gridToggle);
    style.className = 'axis-toggles';
    const maps = Object.keys(COLORMAPS).map(k => [k, k]);
    const mapEl = select(maps, s.colormap, v => { s.colormap = v; this.draw(); });
    mapEl.title = 'Colour style';
    style.append(mapEl);
    style.append(toggle('flip', s.reverse, v => { s.reverse = v; this.controls(); this.draw(); }));

    const save = document.createElement('button');
    save.type = 'button';
    save.className = 'ghost small';
    save.textContent = 'Save image';
    save.addEventListener('click', () => this.saveImage());
    style.append(save);

    const keep = document.createElement('button');
    keep.type = 'button';
    keep.className = 'ghost small';
    keep.textContent = 'Save settings';
    keep.addEventListener('click', () => this.savePreset());
    style.append(keep);
    bar.append(style);

    // The constant lines already added, each with a way to take it off.
    if (s.guides.length) {
      const guides = document.createElement('div');
      guides.className = 'diagram-lines';
      s.guides.forEach((guide, i) => {
        const chip = document.createElement('span');
        chip.className = 'diagram-preset';
        const text = document.createElement('span');
        text.textContent = `${guide.axis} = ${guide.at}`;
        const drop = document.createElement('button');
        drop.type = 'button';
        drop.textContent = '×';
        drop.title = 'Remove this line';
        drop.addEventListener('click', () => {
          s.guides.splice(i, 1); this.controls(); this.draw();
        });
        chip.append(text, drop);
        guides.append(chip);
      });
      this.controlsEl.append(guides);
    }

    if (this.active === 'timeline') this.listLines();

    // Every menu in this bar gets a pair of buttons to step through it with,
    // done here rather than as each is built: they are built detached, and
    // buttons cannot be put either side of an element that has no sides yet.
    for (const menu of bar.querySelectorAll('select')) Metrics.addSteppers(menu);
  },

  /** The series statistics on offer, labelled the way the Viewer labels them. */
  statOptions(runId = this.runId) {
    const payload = this.series.get(runId) || this.series.get(this.runId);
    const keys = payload && payload.keys && payload.keys.length
      ? payload.keys.filter(k => k !== '_frame')
      : Object.keys(Viewer.STAT_LABELS || {});
    return keys.map(k => [k, (Viewer.STAT_LABELS || {})[k] || k]);
  },

  /**
   * One editable row per line.
   *
   * A line is not a label with a delete button — it is the whole set of
   * choices that produced it, still changeable. Which simulation, which
   * statistic, which phase, how many frames of it, and whether it is stretched
   * to the full width so a short run can be compared with a long one by shape.
   */
  listLines() {
    const s = this.now;
    const list = document.createElement('div');
    list.className = 'diagram-lines';

    s.lines.forEach((line, i) => {
      const row = document.createElement('div');
      row.className = 'diagram-line-row';

      const swatch = document.createElement('i');
      swatch.style.background = this.LINE_INK[i % this.LINE_INK.length];
      row.append(swatch);

      const pick = (options, value, onChange, title, live = true) => {
        const el = document.createElement('select');
        for (const [v, text] of options) {
          const option = document.createElement('option');
          option.value = v; option.textContent = text;
          el.append(option);
        }
        el.value = value;
        if (title) el.title = title;
        // Redrawing as the reader moves through a menu is right for a
        // statistic, which is already in hand, and wrong for a simulation,
        // which is a fresh summary each time — arrowing past four runs would
        // start four of them.
        if (live) Metrics.onPick(el, onChange);
        else el.addEventListener('change', () => onChange(el.value));
        row.append(el);
        return el;
      };

      pick(this.runs.map(r => [r.id, r.name]), line.run,
           v => { line.run = v; this.refresh(); }, 'Which simulation', false);
      pick(this.statOptions(line.run), line.stat,
           v => { line.stat = v; this.draw(); }, 'Which statistic');
      // Three buttons rather than a menu, matching the phase control under the
      // Viewer's graph. Three options that are always the same three are worth
      // showing all at once: which one is active is then visible without
      // opening anything, and switching is one click instead of two.
      const phases = document.createElement('span');
      phases.className = 'seg diagram-phase';
      for (const [value, text] of [['all', 'Both phases'],
                                   ['1', 'Reproduction'], ['2', 'Game']]) {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'seg-btn' + (line.phase === value ? ' active' : '');
        button.textContent = text;
        button.title = 'Which phase';
        button.addEventListener('click', () => {
          if (line.phase === value) return;
          line.phase = value;
          for (const other of phases.children) {
            other.classList.toggle('active', other === button);
          }
          // The frame cap's placeholder counts the frames this phase has, so it
          // is stale the moment the phase changes.
          cap.placeholder = String(this.availableFrames(line));
          this.draw();
        });
        phases.append(button);
      }
      row.append(phases);

      // Blank is every frame there is, which is what the number shows when it
      // is left alone rather than an empty box that gives no idea of the scale.
      const cap = document.createElement('input');
      cap.type = 'number';
      cap.min = '2';
      cap.className = 'diagram-cap';
      cap.value = line.maxFrames === null ? '' : String(line.maxFrames);
      cap.placeholder = String(this.availableFrames(line));
      cap.title = 'How many frames of this line to show. Blank is all of them.';
      cap.addEventListener('change', () => {
        line.maxFrames = cap.value === '' ? null : Math.max(2, Number(cap.value));
        this.draw();
      });
      row.append(cap);

      const stretch = document.createElement('button');
      stretch.type = 'button';
      stretch.className = 'axis-btn' + (line.stretch ? ' active' : '');
      stretch.textContent = 'stretch';
      stretch.title = 'Give this line the full width, whatever its length, so runs '
        + 'of different lengths can be compared by shape';
      stretch.addEventListener('click', () => {
        line.stretch = !line.stretch;
        this.controls();
        this.draw();
      });
      row.append(stretch);

      const drop = document.createElement('button');
      drop.type = 'button';
      drop.className = 'diagram-drop';
      drop.textContent = '×';
      drop.title = 'Remove this line';
      drop.addEventListener('click', () => {
        s.lines.splice(i, 1);
        this.controls();
        this.refresh();
      });
      row.append(drop);
      list.append(row);
    });
    this.controlsEl.append(list);
  },

  /** How many frames a line could show, which is what its cap defaults to. */
  availableFrames(line) {
    const payload = this.series.get(line.run);
    if (!payload) return 0;
    const phases = (payload.series || {}).phase || [];
    if (line.phase === 'all') return phases.length;
    return phases.filter(p => String(p) === line.phase).length;
  },
};

// Diagrams is declared in diagrams.js, which the page loads first.
Object.assign(Diagrams, DiagramControls);
