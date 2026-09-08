/*
 * Flow modules over time, drawn.
 *
 * Each module is a band: one segment per iteration it existed, at a height
 * fixed by its identity, thickening with how many agents belonged to it. The
 * segment is lit by its **turnover** — how much of its membership was replaced
 * since the iteration before. A long band that stays dark is a stable group of
 * the same agents; a long band that stays bright is a pattern being carried by
 * different agents each time, which is the interesting one.
 *
 * The arithmetic is all in flowmodules.js, which has its own tests. This is
 * only the fetching and the picture.
 */
const FlowView = {
  runs: [],
  runId: null,
  result: null,

  // How much of a run is on screen at once, in recorded iterations, as a
  // starting value — the reader sets it. It used to read the first 220 frames
  // of any run and offer no way to look further, so a long simulation could
  // only ever be seen starting, and a large one could not be drawn at all.
  MAX_ITERATIONS: 100,

  // The columns the flow arithmetic reads. Everything else in a frame — the
  // edge list above all — is untouched here, and on a large run that is most
  // of what a frame weighs.
  FIELDS: ['iteration', 'phase', 'ids', 'decisions'],

  // Redrawn at most this often while frames are still arriving.
  DRAW_EVERY_MS: 400,

  phase: 'all',

  init() {
    this.canvas = document.getElementById('flowCanvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.noteEl = document.getElementById('flowNote');
    this.readoutEl = document.getElementById('flowReadout');
    this.factsEl = document.getElementById('flowFacts');
    this.floorEl = document.getElementById('flowFloor');
    this.minLifeEl = document.getElementById('flowMinLife');

    // Moving the window refetches, so it acts on release rather than on every
    // pixel of the drag.
    const scrub = document.getElementById('flowWindow');
    scrub.addEventListener('change', () => this.load(this.runId, Number(scrub.value)));

    this.spanEl = document.getElementById('flowSpan');
    this.spanEl.value = String(this.MAX_ITERATIONS);
    this.spanEl.addEventListener('change', () => this.load(this.runId, this.windowStart));

    for (const button of document.querySelectorAll('#flowPhase .seg-btn')) {
      button.addEventListener('click', () => {
        this.phase = button.dataset.phase;
        for (const other of document.querySelectorAll('#flowPhase .seg-btn')) {
          other.classList.toggle('active', other === button);
        }
        this.recompute();
      });
    }

    // The overlap floor changes how modules are matched, so it re-follows the
    // frames it already has rather than fetching them again.
    this.floorEl.addEventListener('input', () => {
      document.getElementById('flowFloorValue').textContent = this.floorEl.value;
    });
    this.floorEl.addEventListener('change', () => this.recompute());
    this.minLifeEl.addEventListener('input', () => {
      document.getElementById('flowMinLifeValue').textContent = this.minLifeEl.value;
      this.draw();
    });

    this.canvas.addEventListener('mousemove', e => this.hover(e));
    this.canvas.addEventListener('mouseleave', () => {
      this.hovered = null;
      this.readoutEl.textContent = '';
      this.draw();
    });

    if (window.ResizeObserver) {
      new ResizeObserver(() => { this.resize(); this.draw(); })
        .observe(this.canvas.parentElement);
    }
  },

  setRuns(runs) {
    this.runs = runs;
  },

  say(text) {
    this.noteEl.textContent = text;
  },

  // ---- reading a run ----------------------------------------------------

  async load(runId, from = null) {
    // Re-entered from the mode switch with nothing new to say: the frames it
    // already holds are the frames it wants, in the place it was left. Moving
    // the window passes a `from`, and so always refetches.
    if (from === null && this.runId === runId && this.frames) {
      this.recompute();
      return;
    }
    const at = from ?? (runId === this.runId ? this.windowStart : 0);
    this.runId = runId;
    this.result = null;
    this.frames = null;
    this.draw();

    const run = this.runs.find(r => r.id === runId);
    if (!run) return;

    // A module's identity is its overlap with the frame before it, so the
    // window is contiguous — a stride would break exactly the thing being
    // measured. It is moved rather than widened.
    const span = Math.max(5, Number(this.spanEl?.value) || this.MAX_ITERATIONS);
    const plan = FrameWindow.plan(run.frame_count, at, span);
    this.windowStart = plan.start;
    this.windowTotal = plan.total;
    FrameWindow.bindScrubber(document.getElementById('flowWindow'), plan);

    const frames = [];
    let painted = 0;
    try {
      for await (const batch of FrameWindow.read(runId, plan.indices, this.FIELDS)) {
        if (this.runId !== runId) return;
        frames.push(...batch);
        this.say(`Reading frames… ${frames.length} of ${plan.indices.length}`);

        // Follow and draw what has arrived rather than waiting out the whole
        // window on a blank canvas.
        const now = Date.now();
        if (now - painted > this.DRAW_EVERY_MS) {
          painted = now;
          this.frames = frames.slice();
          this.recompute();
          this.say(`Reading frames… ${frames.length} of ${plan.indices.length}`);
        }
      }
    } catch (err) {
      this.say(`Could not read the frames: ${err.message}`);
      return;
    }
    this.frames = frames;
    this.recompute();
  },

  recompute() {
    if (!this.frames) return;
    const floor = Number(this.floorEl.value) / 100;
    // A phase filter selects among frames already read rather than changing
    // which are read. Module identity is overlap with the frame before, so
    // filtering to one phase compares each iteration with the last one of the
    // same kind — which is the comparison the reader asked for.
    const kept = this.phase === 'all'
      ? this.frames
      : this.frames.filter(f => String(f.phase) === this.phase);
    if (!kept.length) { this.result = null; this.draw(); return; }
    kept.sort((a, b) => (a.iteration - b.iteration) || (a.phase - b.phase));
    this.result = FlowModules.follow(kept, { floor });
    this.facts = FlowModules.summarise(this.result.history);
    this.resize();
    this.draw();
    this.report();
  },

  report() {
    const { history, withoutFlow } = this.result;
    if (!history.length) {
      this.factsEl.replaceChildren();
      // Tokens are only allocated across links in the game phase, so a
      // reproduction-only filter has nothing to group by construction. Saying
      // "record a run with decisions on" there would be advice that cannot
      // help, about a run that is already recorded correctly.
      this.say(this.phase === '1'
        ? 'Nothing crosses a link during reproduction — tokens are allocated in '
          + 'the game phase, so this view has nothing to group under that filter.'
        : withoutFlow
        ? `None of the ${formatNumber(withoutFlow)} frames read carry what crossed `
          + `each link. Record a run with decisions on and this will have something `
          + `to work from.`
        : 'Nothing to group.');
      return;
    }
    const f = this.facts;
    const cells = [
      ['modules at once', formatNumber(Math.round(f.meanModules)), 'on average'],
      ['compression', `${(f.compression * 100).toFixed(1)}%`, 'shorter than no grouping'],
      ['distinct modules', formatNumber(f.distinct), 'over the whole window'],
      ['longest lived', `${formatNumber(f.longestLife)}`, 'frames'],
      ['largest', formatNumber(f.largest), 'agents'],
      ['turnover', `${(f.meanTurnover * 100).toFixed(0)}%`, 'of members replaced per frame']
    ];
    this.factsEl.replaceChildren(...cells.map(([label, value, hint]) => {
      const cell = document.createElement('div');
      cell.append(
        Object.assign(document.createElement('b'), { textContent: value }),
        Object.assign(document.createElement('i'), { textContent: label }),
        Object.assign(document.createElement('small'), { textContent: hint }));
      return cell;
    }));

    const where = FrameWindow.describe(this.frames, this.windowTotal);
    this.say(`${formatNumber(history.length)} module appearances over `
      + `${formatNumber(new Set(history.map(r => r.iteration)).size)} iterations`
      + (where ? ` — ${where}` : '')
      + (withoutFlow ? `; ${formatNumber(withoutFlow)} frames had no flow recorded.` : '.')
      + (f.compression < 0.01
          ? ' Compression near zero means the flow has no group structure worth'
            + ' the name — whatever is drawn below is one module in all but label.'
          : ''));
  },

  // ---- drawing ----------------------------------------------------------

  resize() {
    const box = this.canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (!box.width || !box.height) return;
    this.canvas.width = Math.round(box.width * dpr);
    this.canvas.height = Math.round(box.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.w = box.width;
    this.h = box.height;
  },

  layout() {
    if (!this.result || !this.w) return null;
    const minLife = Number(this.minLifeEl.value) || 1;

    const byId = new Map();
    for (const row of this.result.history) {
      if (!byId.has(row.id)) byId.set(row.id, []);
      byId.get(row.id).push(row);
    }
    const kept = [...byId.entries()].filter(([, rows]) => rows.length >= minLife);
    if (!kept.length) return null;
    kept.sort((a, b) => a[1][0].iteration - b[1][0].iteration || a[0] - b[0]);

    const iterations = this.result.history.map(r => r.iteration);
    const lo = Math.min(...iterations), hi = Math.max(...iterations);
    const pad = { left: 8, right: 8, top: 10, bottom: 22 };
    const width = this.w - pad.left - pad.right;
    const height = this.h - pad.top - pad.bottom;
    const x = (t) => pad.left + ((t - lo) / Math.max(1, hi - lo)) * width;
    const row = new Map(kept.map(([id], i) => [id, i]));
    const y = (i) => pad.top + ((i + 0.5) / kept.length) * height;
    const thickness = Math.max(1.2, Math.min(9, height / kept.length * 0.72));
    return { kept, row, x, y, pad, width, height, thickness, lo, hi };
  },

  /** Dark when a module keeps its members, bright when it swaps them. */
  colourFor(id, turnover) {
    let h = (id * 2654435761) >>> 0;
    h ^= h >>> 15;
    const hue = (h % 360);
    const light = 34 + Math.min(1, turnover) * 34;
    return `hsl(${hue}, ${48 + Math.min(1, turnover) * 34}%, ${light}%)`;
  },

  draw() {
    if (!this.ctx || !this.w) return;
    const ctx = this.ctx;
    ctx.clearRect(0, 0, this.w, this.h);
    const plan = this.layout();
    if (!plan) {
      ctx.fillStyle = '#6b7c8d';
      ctx.font = '12px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(this.result ? 'Nothing lasted that long.' : 'No run loaded.',
                   this.w / 2, this.h / 2);
      return;
    }

    const { kept, row, x, y, thickness } = plan;
    ctx.lineCap = 'butt';
    for (const [id, rows] of kept) {
      const at = y(row.get(id));
      for (const record of rows) {
        const wide = thickness * (0.4 + Math.min(1.6, Math.log2(1 + record.size) / 3));
        ctx.strokeStyle = this.colourFor(id, record.turnover);
        ctx.lineWidth = id === this.hoveredId ? wide + 2 : wide;
        ctx.beginPath();
        ctx.moveTo(x(record.iteration) - 1, at);
        ctx.lineTo(x(record.iteration) + Math.max(1.5, plan.width / 120), at);
        ctx.stroke();
      }
    }

    ctx.strokeStyle = 'rgba(190, 200, 215, 0.16)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#6b7c8d';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    for (let i = 0; i <= 6; i++) {
      const t = plan.lo + (plan.hi - plan.lo) * (i / 6);
      const at = x(t);
      ctx.beginPath();
      ctx.moveTo(at, plan.pad.top);
      ctx.lineTo(at, plan.pad.top + plan.height);
      ctx.stroke();
      ctx.fillText(String(Math.round(t)), at, this.h - 7);
    }
  },

  hover(event) {
    const plan = this.layout();
    if (!plan) return;
    const box = this.canvas.getBoundingClientRect();
    const my = event.clientY - box.top;
    let found = null, gap = Infinity;
    for (const [id] of plan.kept) {
      const at = plan.y(plan.row.get(id));
      if (Math.abs(at - my) < gap) { gap = Math.abs(at - my); found = id; }
    }
    if (gap > 8) found = null;
    if (found === this.hoveredId) return;
    this.hoveredId = found;

    if (found === null) {
      this.readoutEl.textContent = '';
    } else {
      const rows = plan.kept.find(([id]) => id === found)[1];
      const churn = rows.slice(1).reduce((s, r) => s + r.turnover, 0) / Math.max(1, rows.length - 1);
      const sizes = rows.map(r => r.size);
      this.readoutEl.textContent =
        `module ${found} — ${rows.length} frames, from iteration ${rows[0].iteration} `
        + `to ${rows[rows.length - 1].iteration}, ${Math.min(...sizes)}–${Math.max(...sizes)} `
        + `agents, ${(churn * 100).toFixed(0)}% of its members replaced per frame`;
    }
    this.draw();
  }
};
