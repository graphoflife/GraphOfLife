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
  // edge list above all, and the winners and pruned edges beside the
  // allocations — is untouched here, and on a large run that is most of what a
  // frame weighs: 129 MB for 24 frames whole, against a fraction of that for
  // these four.
  FIELDS: ['iteration', 'phase', 'ids', 'decisions.allocations'],

  // Stop reading once this many agent-sightings have arrived. A window
  // measured in iterations is a very different amount of work in a world of
  // sixty agents and one of forty thousand, and clustering a frame is
  // superlinear in the world — so the window has to be bounded by the world,
  // not only by the reader's patience.
  MAX_SIGHTINGS: 120000,

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
    let seen = 0;
    try {
      for await (const batch of FrameWindow.read(runId, plan.indices, this.FIELDS,
                                                 this.MAX_SIGHTINGS)) {
        if (this.runId !== runId) return;
        frames.push(...batch);
        seen += batch.reduce((n, f) => n + (f.ids || []).length, 0);
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
        // Enough of this world seen. Reading further would add minutes of
        // clustering for a picture already at the limit of what can be read.
        if (seen >= this.MAX_SIGHTINGS) break;
      }
    } catch (err) {
      this.say(`Could not read the frames: ${err.message}`);
      return;
    }
    this.frames = frames;
    this.asked = plan.indices.length;
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
    // Clustering a frame costs the better part of a second on a large world, so
    // this view is bounded by the world rather than by the number in the box.
    // Said outright: a control that is overruled in silence is worse than none.
    const short = this.asked && this.frames.length < this.asked
      ? ` — ${formatNumber(this.frames.length)} frames of the `
        + `${formatNumber(this.asked)} asked for, which is as much of a world `
        + `this size as can be clustered in reasonable time`
      : '';
    this.say(`${formatNumber(history.length)} module appearances over `
      + `${formatNumber(new Set(history.map(r => r.iteration)).size)} iterations`
      + short
      + (where && !short ? ` — ${where}` : '')
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

  /**
   * The columns, the blocks in each, and the ribbons between them.
   *
   * An alluvial diagram, which is what Rosvall & Bergstrom built to show how
   * the map equation's modules change over time — the same method this view
   * runs. It replaces one horizontal band per module, which needed a long run
   * of iterations to read as anything and could not have one: clustering a
   * frame of a large world costs the better part of a second, so this view
   * gets a dozen iterations, not a hundred. A dozen columns of blocks joined by
   * ribbons says more about splitting and merging than a hundred hairlines
   * would.
   *
   * The ribbons are not estimated. A history row already carries `born` and
   * `joined` — everyone who was not in that module last time — so the agents
   * carried over from its predecessor are `size - born - joined`, exactly.
   */
  layout() {
    if (!this.result || !this.w) return null;
    const minLife = Number(this.minLifeEl.value) || 1;

    const lives = new Map();
    for (const row of this.result.history) {
      if (!lives.has(row.id)) lives.set(row.id, 0);
      lives.set(row.id, lives.get(row.id) + 1);
    }

    // One column per recorded frame, in order.
    const columns = [];
    const seen = new Map();
    for (const row of this.result.history) {
      const at = `${row.iteration}:${row.phase}`;
      let column = seen.get(at);
      if (!column) {
        column = { iteration: row.iteration, phase: row.phase, blocks: [], total: 0 };
        seen.set(at, column);
        columns.push(column);
      }
      if (lives.get(row.id) < minLife) continue;
      column.blocks.push(row);
      column.total += row.size;
    }
    columns.sort((a, b) => (a.iteration - b.iteration) || (a.phase - b.phase));
    if (columns.length < 2 || !columns.some(c => c.total > 0)) return null;

    // Ordered so ribbons cross as little as possible: the first column by size,
    // and every column after it by where its predecessor sat. A module with no
    // predecessor is new and goes to the end.
    let rank = new Map();
    columns.forEach((column, index) => {
      if (index === 0) {
        column.blocks.sort((a, b) => b.size - a.size);
      } else {
        const previous = rank;
        column.blocks.sort((a, b) =>
          (previous.has(a.id) ? previous.get(a.id) : Infinity)
          - (previous.has(b.id) ? previous.get(b.id) : Infinity)
          || b.size - a.size);
      }
      rank = new Map(column.blocks.map((block, i) => [block.id, i]));
    });

    const pad = { left: 10, right: 10, top: 12, bottom: 24 };
    const width = this.w - pad.left - pad.right;
    const height = this.h - pad.top - pad.bottom;
    const columnWidth = Math.max(6, Math.min(26, width / (columns.length * 2.6)));
    const at = (i) => pad.left + (columns.length < 2 ? 0.5 : i / (columns.length - 1))
      * (width - columnWidth);

    // A gap between blocks, so a column reads as several modules rather than
    // one bar — but never so much that the blocks vanish.
    for (const column of columns) {
      const gaps = Math.max(0, column.blocks.length - 1);
      const gap = Math.min(2, (height * 0.25) / Math.max(1, gaps));
      const usable = Math.max(1, height - gap * gaps);
      const scale = column.total > 0 ? usable / column.total : 0;
      let y = pad.top;
      for (const block of column.blocks) {
        block.top = y;
        block.height = block.size * scale;
        y += block.height + gap;
      }
    }

    return { columns, at, columnWidth, pad, width, height };
  },

  /** A module's colour, from its identity, so it keeps it across columns. */
  colourFor(id, turnover) {
    const hue = (Math.imul(id ^ 0x9e3779b9, 2654435761) >>> 0) % 360;
    // Brightness is turnover: a band that stays dark is the same agents, a
    // bright one is a pattern being carried by different matter each time.
    const light = 38 + Math.min(1, turnover || 0) * 34;
    return `hsl(${hue}, 62%, ${light}%)`;
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
      ctx.textAlign = 'left';
      return;
    }

    const { columns, at, columnWidth } = plan;

    // The ribbons first, underneath the blocks, so a block's edge stays crisp.
    for (let c = 0; c + 1 < columns.length; c++) {
      const here = new Map(columns[c].blocks.map(b => [b.id, b]));
      const leftX = at(c) + columnWidth;
      const rightX = at(c + 1);

      // Stacked at each end in the same order the blocks are, so several
      // ribbons leaving one module do not overlap each other.
      const usedFrom = new Map(), usedTo = new Map();
      for (const block of columns[c + 1].blocks) {
        const from = here.get(block.id);
        if (!from) continue;
        const carried = Math.max(0, block.size - block.born - block.joined);
        if (carried <= 0) continue;
        const scaleFrom = from.height / Math.max(1, from.size);
        const scaleTo = block.height / Math.max(1, block.size);
        const a0 = from.top + (usedFrom.get(from.id) || 0);
        const a1 = a0 + carried * scaleFrom;
        const b0 = block.top + (usedTo.get(block.id) || 0);
        const b1 = b0 + carried * scaleTo;
        usedFrom.set(from.id, (usedFrom.get(from.id) || 0) + carried * scaleFrom);
        usedTo.set(block.id, (usedTo.get(block.id) || 0) + carried * scaleTo);

        const mid = (leftX + rightX) / 2;
        ctx.beginPath();
        ctx.moveTo(leftX, a0);
        ctx.bezierCurveTo(mid, a0, mid, b0, rightX, b0);
        ctx.lineTo(rightX, b1);
        ctx.bezierCurveTo(mid, b1, mid, a1, leftX, a1);
        ctx.closePath();
        ctx.fillStyle = this.colourFor(block.id, block.turnover);
        ctx.globalAlpha = block.id === this.hoveredId ? 0.55 : 0.22;
        ctx.fill();
        ctx.globalAlpha = 1;
      }
    }

    for (let c = 0; c < columns.length; c++) {
      for (const block of columns[c].blocks) {
        ctx.fillStyle = this.colourFor(block.id, block.turnover);
        ctx.globalAlpha = block.id === this.hoveredId ? 1 : 0.92;
        ctx.fillRect(at(c), block.top, columnWidth, Math.max(1, block.height));
        ctx.globalAlpha = 1;
      }
    }

    ctx.fillStyle = '#8fa3b5';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    const step = Math.max(1, Math.round(columns.length / 8));
    for (let c = 0; c < columns.length; c += step) {
      ctx.fillText(String(columns[c].iteration), at(c) + columnWidth / 2, this.h - 7);
    }
    ctx.textAlign = 'left';
  },

  hover(event) {
    const plan = this.layout();
    if (!plan) return;
    const box = this.canvas.getBoundingClientRect();
    const mx = event.clientX - box.left, my = event.clientY - box.top;

    // Which block the pointer is inside. A hit test, since a block has an area.
    let found = null, row = null;
    for (let c = 0; c < plan.columns.length; c++) {
      const left = plan.at(c);
      if (mx < left - 3 || mx > left + plan.columnWidth + 3) continue;
      for (const block of plan.columns[c].blocks) {
        if (my >= block.top && my <= block.top + Math.max(2, block.height)) {
          found = block.id; row = block; break;
        }
      }
      break;
    }
    if (found === this.hoveredId) return;
    this.hoveredId = found;
    this.readoutEl.textContent = row
      ? `module ${row.id} at iteration ${row.iteration} — ${formatNumber(row.size)} agents, `
        + `${Math.round(row.turnover * 100)}% of them new since the frame before `
        + `(${formatNumber(row.born)} born, ${formatNumber(row.joined)} moved in, `
        + `${formatNumber(row.left)} moved out, ${formatNumber(row.died)} gone)`
      : '';
    this.draw();
  }
};
