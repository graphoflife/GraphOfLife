/*
 * One panel per testable claim about the graph.
 *
 * Each entry states the claim, the mechanism that would make it true, and —
 * written down *before* the run is looked at — what would confirm it and what
 * would refute it. Then it plots the series that decide it, for whichever
 * simulation is selected.
 *
 * Deliberately no conclusions. A panel that told you the answer would be doing
 * the arguing that the experiment is supposed to do, and this project has twice
 * had a number that looked like a finding until it was put next to a control
 * (conquest cycles ran *below* chance). The prose says what to look for; the
 * reader looks.
 *
 * THESES is data. A new claim is one entry — a title, four paragraphs and a
 * list of series to draw — and nothing else in this file changes.
 */
const Theses = {

  /**
   * `series` names keys in the run's series rows. `derive` builds a series that
   * is not stored directly, from the whole row. `rule` draws a horizontal line
   * at a value the claim is stated in terms of.
   */
  THESES: [
    {
      id: 'shortcut',
      title: 'The shortcut budget is spent',
      claim: 'The graph starts as a small world and cannot stay one. Its '
        + 'long-range edges are a one-time endowment that can only be lost.',
      why: 'Every edge the engine creates joins nodes at most two hops apart: a '
        + 'newborn links into its parent’s neighbourhood, and a handover '
        + 'moves an edge within it. Nothing anywhere creates a link between '
        + 'distant parts of the graph. Meanwhile edges are destroyed freely — '
        + 'every edge carrying no tokens is cut at the end of each Blotto phase. '
        + 'The Watts–Strogatz rewiring the world is built with is therefore '
        + 'a budget that is spent and never refilled.',
      confirm: 'Bridges rise as a share of edges, loop density falls, and the '
        + 'graph sits far from where a Watts–Strogatz graph of the same size '
        + 'and degree would be.',
      refute: 'The measures hold near their starting values, or return toward '
        + 'them after an early excursion — which would mean something is '
        + 'replacing long-range structure that this reading of the code says '
        + 'cannot be replaced.',
      series: [
        { key: 'loopDensity', label: 'Loop density', colour: '#5ac8fa' },
        { key: 'transitivity', label: 'Clustering', colour: '#ffd166' },
        { derive: r => (r.edges && r.bridges != null ? r.bridges / r.edges : null),
          label: 'Bridges / edges', colour: '#ff6b6b' }
      ]
    },
    {
      id: 'fragility',
      title: 'Fragility leads the cull',
      claim: 'How lopsided the graph is predicts how many agents the next '
        + 'cleanup removes.',
      why: 'Cleanup keeps only the largest connected component and kills '
        + 'everything else. So a bridge with a tenth of the population behind it '
        + 'is not a curiosity — it is a tenth-of-the-population extinction, '
        + 'waiting for the zero-flow prune to happen to cut that one edge. If '
        + 'that is what is going on, the worst cut available *before* a phase '
        + 'should say something about the cull that phase produces.',
      confirm: 'Worst cut, taken before the cull, moves ahead of the culled '
        + 'share — and keeps doing so against a control that shifts one '
        + 'series in time and destroys only the alignment between them.',
      refute: 'No relationship beyond the shifted control, which would mean the '
        + 'culls are removing stragglers rather than severed regions, and the '
        + 'bridge structure is not what kills anyone.',
      note: 'The pre-cull reading is what makes this answerable at all. The '
        + 'ordinary bridge count is taken after cleanup, in the same frame as the '
        + 'cull, so a cut that severs a whole side moves both numbers at once and '
        + 'cause cannot be told from effect. Runs recorded before the engine '
        + 'started taking the pre-cull reading cannot test this.',
      series: [
        { key: 'cutRiskBefore', label: 'Worst cut, before the cull', colour: '#ff6b6b' },
        { derive: r => (r.nodes_before && r.orphaned != null
                        ? r.orphaned / r.nodes_before : null),
          label: 'Culled share', colour: '#5ac8fa' }
      ]
    },
    {
      id: 'lopsided',
      title: 'No cut costs more than a tenth',
      claim: 'Squash every redundant blob to a point and what is left is a tree '
        + 'whose edges are exactly the bridges. The claim is about how lopsided '
        + 'that tree is: no single bridge has more than a tenth of the '
        + 'population on one side of it.',
      why: 'That is the shape of a decentralised network — plenty of small '
        + 'tree-like fringes hanging off local points, but no chokepoint whose '
        + 'loss splits the world in half. It is also what real social and '
        + 'information networks look like: a well-connected core that no cut '
        + 'divides cheaply, with small pieces attached by single edges. If this '
        + 'population is organising rather than merely growing, this is the '
        + 'shape it would organise into.',
      confirm: 'Worst cut stays under the line and does not trend toward it. '
        + 'The population is a core with fringes, and no one edge holds it '
        + 'together.',
      refute: 'Worst cut climbs toward 50%, meaning the graph is a dumbbell held '
        + 'by one edge and every measurement of “organisation” is really '
        + 'a measurement of which half survived.',
      rule: { at: 0.10, label: 'the tenth' },
      series: [
        { key: 'cutRisk', label: 'Worst cut', colour: '#ff6b6b' }
      ]
    },
    {
      id: 'curvature',
      title: 'The graph is negatively curved',
      claim: 'Neighbourhoods hold more than flat space allows, and keep doing so.',
      why: 'The dimension estimate walks outward from a node and fits how fast '
        + 'the frontier grows. In curved space that line bends, and the bend is '
        + 'the Ricci scalar rather than noise — the same fit gives both, and '
        + 'until now the second half was being discarded as residual. The sign '
        + 'is the interesting part: a branching, tree-like graph is strongly '
        + 'negative, and bounded-degree expanders are negatively curved as a '
        + 'theorem, so this doubles as a reading on how expander-like the graph '
        + 'is.',
      confirm: 'Curvature is negative and stable, or drifts steadily — either '
        + 'is a geometry rather than an artefact.',
      refute: 'It swings wildly frame to frame, in which case there are too few '
        + 'usable radii for the fit and the number is measuring the sampling, '
        + 'not the graph.',
      series: [
        { key: 'ricciCurvature', label: 'Ricci curvature', colour: '#c792ea' },
        { key: 'dimension', label: 'Dimension', colour: '#5ac8fa' }
      ]
    },
    {
      id: 'core',
      title: 'A core with whiskers',
      claim: 'The population splits into a well-connected core and a tree-shaped '
        + 'fringe hanging off it, rather than being uniform.',
      why: 'Peel away everyone with a single connection, and keep peeling until '
        + 'nobody has one. What survives is the part where every agent sits on '
        + 'some loop; what is peeled is fringe. Real networks are overwhelmingly '
        + 'this shape, and no standard random-graph model reproduces it, so '
        + 'finding it here would not be a property inherited from the starting '
        + 'graph.',
      confirm: 'Core share settles well below 1 and stays there — a stable '
        + 'division of labour between core and fringe rather than a transient.',
      refute: 'Core share sits near 1 (no fringe, the graph is uniformly '
        + 'interwoven) or falls toward 0 (no core, the graph is becoming a tree).',
      series: [
        { key: 'coreShare', label: 'Core share', colour: '#7ee787' },
        { derive: r => (r.nodes && r.leaves != null ? r.leaves / r.nodes : null),
          label: 'Leaf share', colour: '#ffd166' }
      ]
    }
  ],

  runs: [],
  runId: null,
  rows: null,
  active: 'shortcut',

  get thesis() {
    return this.THESES.find(t => t.id === this.active) || this.THESES[0];
  },

  init() {
    this.canvas = document.getElementById('thesesCanvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.noteEl = document.getElementById('thesesNote');
    this.proseEl = document.getElementById('thesesProse');
    this.listEl = document.getElementById('thesesList');
    this.legendEl = document.getElementById('thesesLegend');

    this.listEl.replaceChildren(...this.THESES.map(t => {
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = t.title;
      button.dataset.thesis = t.id;
      button.addEventListener('click', () => this.select(t.id));
      return button;
    }));
    this.select(this.active);
  },

  setRuns(runs) { this.runs = runs; },

  say(text) { if (this.noteEl) this.noteEl.textContent = text; },

  select(id) {
    this.active = id;
    for (const button of this.listEl.querySelectorAll('button')) {
      button.classList.toggle('active', button.dataset.thesis === id);
    }
    this.paint();
    this.resize();
    this.draw();
  },

  /** The written half. Rebuilt on every selection; it is four paragraphs. */
  paint() {
    const t = this.thesis;
    const block = (label, body, cls) => {
      const p = document.createElement('p');
      if (cls) p.className = cls;
      const b = document.createElement('b');
      b.textContent = `${label} `;
      p.append(b, document.createTextNode(body));
      return p;
    };
    const head = document.createElement('h3');
    head.textContent = t.title;

    const parts = [
      head,
      block('The claim.', t.claim),
      block('Why it might be true.', t.why),
      block('Confirmed by.', t.confirm),
      block('Refuted by.', t.refute)
    ];
    if (t.note) parts.push(block('Note.', t.note, 'theses-note'));
    this.proseEl.replaceChildren(...parts);
  },

  /** Columns to rows — a claim can be about a quantity that is not stored. */
  absorb(data) {
    const columns = data.series || {};
    const length = (columns.iteration || []).length;
    this.rows = Array.from({ length }, (_, i) => {
      const row = {};
      for (const key of data.keys || []) row[key] = columns[key][i];
      return row;
    });
    this.strain = data.strain || null;
    this.resize();
    this.draw();
  },

  async load(runId) {
    // Each load supersedes the one before it, so a climb that is still going
    // when the run changes stops drawing into the new one's chart.
    const token = (this.token = (this.token || 0) + 1);
    // What the tab was showing when this started. Checked between requests, so
    // walking away stops the work rather than leaving it running unseen.
    const epoch = Research.epoch;
    this.runId = runId;
    this.rows = null;
    this.strain = null;

    if (!runId) {
      this.say('Choose a simulation above to plot this against.');
      this.draw();
      return;
    }

    this.say('Reading the run…');
    this.draw();
    try {
      await SeriesLoad.climb(runId, {
        cancelled: () => this.token !== token || Research.stale(epoch),
        onStep: (data, done) => {
          this.absorb(data);
          const at = SeriesLoad.fraction(data);
          this.say(done || data.complete
            ? `${formatNumber(data.points || this.rows.length)} points`
              + (data.sampled ? `, every ${data.stride} iterations` : '')
              + (this.strain ? ` — algorithm ${this.strain}` : '')
            : `Refining… ${formatNumber(data.points || 0)} of `
              + `${formatNumber(data.totalPoints || 0)} points`
              + (at !== null ? ` (${Math.round(at * 100)}%)` : ''));
        }
      });
    } catch (err) {
      if (this.token !== token) return;
      this.rows = null;
      this.say(`Could not read the series: ${err.message}`);
      this.draw();
    }
  },

  resize() {
    if (!this.canvas) return;
    const box = this.canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    // A hidden canvas measures zero, and a zero-sized canvas throws away
    // whatever was drawn into it. Keep the last good size until there is a
    // real one to replace it with.
    if (box.width < 1 || box.height < 1) return;
    this.canvas.width = Math.round(box.width * dpr);
    this.canvas.height = Math.round(box.height * dpr);
    this.canvas.style.width = `${box.width}px`;
    this.canvas.style.height = `${box.height}px`;
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.width = box.width;
    this.height = box.height;
  },

  /**
   * One series' values against iteration, with the gaps kept as gaps.
   *
   * A missing key is not zero — a run recorded before a measurement existed has
   * nothing to say about it, and drawing that as a flat line at the bottom
   * would be inventing data. Nulls break the line instead.
   *
   * Which is also why every `derive` above guards its inputs explicitly. In
   * JavaScript `null / 60` is 0, not null, and `Number.isFinite(0)` is true,
   * so a derived series turned an absent measurement into a confident zero
   * and drew it — the exact failure this note is about, arriving through the
   * one path the check could not see.
   */
  extract(spec) {
    const points = [];
    let present = 0;
    for (const row of this.rows) {
      const value = spec.derive ? spec.derive(row) : row[spec.key];
      const ok = value !== null && value !== undefined && Number.isFinite(value);
      if (ok) present++;
      points.push({ x: row.iteration, y: ok ? value : null });
    }
    return { spec, points, present };
  },

  draw() {
    if (!this.ctx) return;
    const { ctx } = this;
    const w = this.width || 0, h = this.height || 0;
    ctx.clearRect(0, 0, w, h);
    if (!w || !h) return;

    if (!this.rows || !this.rows.length) {
      this.legendEl.replaceChildren();
      ctx.fillStyle = 'rgba(255,255,255,0.4)';
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(this.runId ? 'Reading the run…' : 'No simulation chosen.',
                   w / 2, h / 2);
      ctx.textAlign = 'left';
      return;
    }

    const t = this.thesis;
    const tracks = t.series.map(s => this.extract(s));
    const drawable = tracks.filter(track => track.present > 1);

    if (!drawable.length) {
      this.legendEl.replaceChildren();
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.font = '13px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('This simulation does not carry the measurements this claim needs.',
                   w / 2, h / 2 - 9);
      ctx.fillText('It was recorded before they existed — a new run would have them.',
                   w / 2, h / 2 + 11);
      ctx.textAlign = 'left';
      return;
    }

    const pad = { left: 46, right: 12, top: 12, bottom: 26 };
    const plotW = Math.max(1, w - pad.left - pad.right);
    const plotH = Math.max(1, h - pad.top - pad.bottom);

    const xs = this.rows.map(r => r.iteration);
    const xMin = Math.min(...xs), xMax = Math.max(...xs);
    const xAt = v => pad.left + (xMax > xMin ? (v - xMin) / (xMax - xMin) : 0.5) * plotW;

    // Every series is drawn against its own range, because these are different
    // quantities in different units and forcing them onto one axis would only
    // flatten whichever has the smaller spread. The legend carries each range,
    // so the shape is comparable even though the scale is not.
    for (const track of drawable) {
      const values = track.points.filter(p => p.y !== null).map(p => p.y);
      track.lo = Math.min(...values);
      track.hi = Math.max(...values);
      if (track.hi === track.lo) { track.hi = track.lo + 1; }
      // A rule is stated on the same scale as the series it belongs to, so it
      // has to be inside the range or it cannot be drawn where it means.
      if (t.rule && drawable.length === 1) {
        track.lo = Math.min(track.lo, 0);
        track.hi = Math.max(track.hi, t.rule.at * 1.1);
      }
      track.yAt = v => pad.top + (1 - (v - track.lo) / (track.hi - track.lo)) * plotH;
    }

    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.strokeRect(pad.left + 0.5, pad.top + 0.5, plotW, plotH);

    if (t.rule && drawable.length === 1) {
      const track = drawable[0];
      const y = track.yAt(t.rule.at);
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = 'rgba(255,255,255,0.45)';
      ctx.beginPath();
      ctx.moveTo(pad.left, y);
      ctx.lineTo(pad.left + plotW, y);
      ctx.stroke();
      ctx.restore();
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.font = '11px system-ui, sans-serif';
      ctx.fillText(t.rule.label, pad.left + 4, y - 4);
    }

    for (const track of drawable) {
      ctx.strokeStyle = track.spec.colour;
      ctx.lineWidth = 1.4;
      ctx.beginPath();
      let down = true;
      for (const point of track.points) {
        if (point.y === null) { down = true; continue; }
        const x = xAt(point.x), y = track.yAt(point.y);
        if (down) { ctx.moveTo(x, y); down = false; } else { ctx.lineTo(x, y); }
      }
      ctx.stroke();
    }

    ctx.fillStyle = 'rgba(255,255,255,0.5)';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(`${formatNumber(xMin)}`, pad.left, h - 9);
    ctx.textAlign = 'right';
    ctx.fillText(`${formatNumber(xMax)} iterations`, pad.left + plotW, h - 9);
    ctx.textAlign = 'left';

    const round = v => (Math.abs(v) >= 100 ? v.toFixed(0)
                      : Math.abs(v) >= 1 ? v.toFixed(2) : v.toFixed(4));
    this.legendEl.replaceChildren(...drawable.map(track => {
      const item = document.createElement('span');
      item.className = 'theses-key';
      const swatch = document.createElement('i');
      swatch.style.background = track.spec.colour;
      item.append(swatch, document.createTextNode(
        `${track.spec.label} — ${round(track.lo)} to ${round(track.hi)}`));
      return item;
    }));
  }
};
