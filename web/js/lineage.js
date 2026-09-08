/*
 * The lineage forest of a run, drawn.
 *
 * Every genotype a run ever held is a horizontal line, running from the first
 * frame it appeared in to the last, thickening with how many agents carried it.
 * A vertical stroke joins each one to the genotype it mutated from. Clades sit
 * together because the vertical order is a depth-first walk of the forest, so a
 * lineage and all its descendants occupy one contiguous band.
 *
 * Colour is the ancestry. A founder is given a colour at random; every mutation
 * shifts its parent's colour slightly. So relatedness is visible directly —
 * a band of one hue is one family, and a run where everything ends up the same
 * colour is a run where one founder's descendants took the world.
 *
 * This reads the frames a run recorded, which is only possible because a brain
 * id names a genotype: a copy keeps its source's id and only mutation makes a
 * new one. While an id was handed out per copy, the id linking one recorded
 * brain to the next was itself never recorded, and half of what a run created
 * never reached a frame at all.
 */
const Lineage = {
  runs: [],
  runId: null,
  forest: null,
  active: false,

  // How much of a run is on screen at once, as a starting value — the reader
  // sets it, since how far you want to see at once depends on what you are
  // looking for. The window is contiguous and you move it with the slider; see
  // framewindow.js for why it cannot simply be sampled instead.
  MAX_ITERATIONS: 100,

  // The columns a lineage actually reads. A frame of a large run is mostly its
  // edge list and this draws none of it, so asking for whole frames was tens of
  // megabytes parsed per window to get at two arrays.
  FIELDS: ['iteration', 'phase', 'brain_ids', 'parent_brain_ids'],

  // Redrawn at most this often while frames are still arriving. Rebuilding the
  // forest on every batch is wasted work at this size; rebuilding it never
  // until the last frame lands is an empty box for the whole read.
  DRAW_EVERY_MS: 250,

  phase: 'all',
  // Frames times agents, roughly. A big world is tens of thousands of agents a
  // frame, and a hundred iterations of those is millions of genotypes to lay
  // out and a picture nobody can read — so the window shortens itself once the
  // first batch has said how big this world is.
  MAX_SIGHTINGS: 400000,
  // Above this share of parentless genotypes the run predates brain ids naming
  // a genotype, and its genealogy cannot be rebuilt from what it recorded.
  ROOTS_SUSPECT: 0.25,

  init() {
    this.canvas = document.getElementById('lineageCanvas');
    if (!this.canvas) return;
    this.ctx = this.canvas.getContext('2d');
    this.noteEl = document.getElementById('lineageNote');
    this.minLifeEl = document.getElementById('lineageMinLife');
    this.readoutEl = document.getElementById('lineageReadout');

    this.minLifeEl.addEventListener('input', () => {
      document.getElementById('lineageMinLifeValue').textContent = this.minLifeEl.value;
      this.draw();
    });
    // Moving the window refetches, so it acts on release rather than on every
    // pixel of the drag.
    const scrub = document.getElementById('lineageWindow');
    scrub.addEventListener('change', () => this.load(this.runId, Number(scrub.value)));

    // How many iterations are on screen at once. Changing it refetches, so it
    // acts on commit rather than on every keystroke.
    this.spanEl = document.getElementById('lineageSpan');
    this.spanEl.value = String(this.MAX_ITERATIONS);
    this.spanEl.addEventListener('change', () => this.load(this.runId, this.windowStart));

    for (const button of document.querySelectorAll('#lineagePhase .seg-btn')) {
      button.addEventListener('click', () => {
        this.phase = button.dataset.phase;
        for (const other of document.querySelectorAll('#lineagePhase .seg-btn')) {
          other.classList.toggle('active', other === button);
        }
        // A phase filter is a filter on frames already in hand, so it redraws
        // from them rather than reading the window again.
        this.rebuild();
      });
    }

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

  /** The tab owns the list; this view just needs to know what is in it. */
  setRuns(runs) {
    this.runs = runs;
  },

  /**
   * Rebuild the forest from the frames in hand and draw it.
   *
   * Separate from `load` because two things call it: each batch as it arrives,
   * and the phase toggle, which changes which of the frames already read are
   * counted rather than which frames are read.
   */
  rebuild(reading = false) {
    if (!this.frames) return;
    const kept = this.phase === 'all'
      ? this.frames
      : this.frames.filter(f => String(f.phase) === this.phase);
    kept.sort((a, b) => (a.iteration - b.iteration) || (a.phase - b.phase));
    this.forest = kept.length ? this.build(kept) : null;
    this.resize();
    this.draw();
    if (!reading) this.describe(kept);
  },

  /**
   * Say what is on screen.
   *
   * Written here rather than at the end of the read, because the phase filter
   * changes the picture without reading anything — and a note left over from
   * the unfiltered forest described a drawing that was no longer there.
   */
  describe(frames) {
    const f = this.forest;
    if (!f) {
      this.say('No frames of that phase in this window.');
      return;
    }
    const plan = this.plan || { total: frames.length, start: 0 };
    const where = FrameWindow.describe(frames, plan.total);
    const rootShare = f.roots.length / Math.max(1, f.order.length);

    let note = `${formatNumber(f.order.length)} genotypes over `
      + `${formatNumber(f.lastIteration - f.firstIteration + 1)} iterations, `
      + `${formatNumber(f.roots.length)} without a parent inside the window`
      + (where
          ? ` — ${where}, so those are ancestors from before it.`
          : `, which are its founders.`);

    // A run recorded before a brain id named a genotype hands out a fresh id
    // on every copy and mutates it away in the same phase, so the id linking
    // one recorded brain to the next was never itself recorded. Half of every
    // chain is missing and no picture drawn from it means anything.
    if (rootShare > this.ROOTS_SUSPECT && plan.start === 0) {
      note += ` That is ${(rootShare * 100).toFixed(0)}% of them, which is too`
        + ` many to be founders: this run was recorded before a brain id named`
        + ` a genotype, so its ancestry cannot be rebuilt. Run a new simulation`
        + ` to see a real one.`;
    }
    this.say(note);
  },

  say(text) {
    this.noteEl.textContent = text;
  },

  // ---- reading a run ----------------------------------------------------

  async load(runId, from = null) {
    // Coming back to the same run — a mode switch, a refresh — reopens where
    // it was left rather than snapping to the beginning. A different run has
    // no remembered place, so it starts at its own.
    const at = from ?? (runId === this.runId ? this.windowStart : 0);
    this.runId = runId;
    this.forest = null;
    this.frames = null;
    this.draw();

    const run = this.runs.find(r => r.id === runId);
    if (!run) return;

    const span = Math.max(5, Number(this.spanEl?.value) || this.MAX_ITERATIONS);
    const plan = FrameWindow.plan(run.frame_count, at, span);
    this.plan = plan;
    this.windowStart = plan.start;
    FrameWindow.bindScrubber(document.getElementById('lineageWindow'), plan);

    const frames = [];
    let budget = plan.indices.length;
    let painted = 0;
    try {
      for await (const batch of FrameWindow.read(runId, plan.indices, this.FIELDS)) {
        if (this.runId !== runId) return;          // a different run was picked
        frames.push(...batch);

        // Now that the size of the world is known, take only as many frames as
        // will make a picture rather than a wall.
        if (frames.length === batch.length) {
          const agents = frames[0].brain_ids.length || 1;
          budget = Math.max(batch.length,
                            Math.min(budget, Math.ceil(this.MAX_SIGHTINGS / agents)));
        }
        this.say(`Reading frames… ${frames.length} of ${budget}`);

        // Draw what has arrived. A window of a large run takes seconds to read,
        // and an empty box for all of it reads as broken rather than as busy.
        const now = Date.now();
        if (now - painted > this.DRAW_EVERY_MS) {
          painted = now;
          this.frames = frames.slice();
          this.rebuild(true);
          this.say(`Reading frames… ${frames.length} of ${budget}`);
        }
        if (frames.length >= budget) break;
      }
    } catch (err) {
      this.say(`Could not read the frames: ${err.message}`);
      return;
    }

    this.frames = frames;
    this.rebuild();
  },

  /**
   * Frames in, forest out.
   *
   * A genotype is a node; its parent is whatever the frame said the first time
   * it was seen. Anything whose parent is not itself inside the window becomes
   * a root: reading a whole run those are its founders, and reading a window
   * they are the ancestors it inherited from before the window began.
   */
  build(frames) {
    const nodes = new Map();
    let firstIteration = Infinity;
    let lastIteration = -Infinity;

    for (const frame of frames) {
      const t = frame.iteration;
      firstIteration = Math.min(firstIteration, t);
      lastIteration = Math.max(lastIteration, t);
      const ids = frame.brain_ids || [];
      const parents = frame.parent_brain_ids || [];

      const here = new Map();
      for (let i = 0; i < ids.length; i++) {
        here.set(ids[i], (here.get(ids[i]) || 0) + 1);
      }
      for (let i = 0; i < ids.length; i++) {
        const id = ids[i];
        let node = nodes.get(id);
        if (!node) {
          node = { id, parent: parents[i], born: t, died: t, peak: 0, span: 0 };
          nodes.set(id, node);
        }
        node.died = t;
      }
      for (const [id, count] of here) {
        const node = nodes.get(id);
        node.peak = Math.max(node.peak, count);
        node.span += 1;
      }
    }

    // Children, in the order they appeared, so a clade reads left to right.
    const children = new Map();
    const roots = [];
    for (const node of nodes.values()) {
      const parent = nodes.get(node.parent);
      if (parent && parent !== node) {
        if (!children.has(parent.id)) children.set(parent.id, []);
        children.get(parent.id).push(node);
      } else {
        roots.push(node);
      }
    }
    for (const list of children.values()) list.sort((a, b) => a.born - b.born);
    roots.sort((a, b) => a.born - b.born);

    // Depth-first, so every lineage and its descendants form one band, and
    // colour, which is inherited with a nudge at each mutation.
    const order = [];
    for (const root of roots) {
      const stack = [[root, this.founderColour(root.id), 0]];
      while (stack.length) {
        const [node, colour, depth] = stack.pop();
        node.colour = colour;
        node.depth = depth;
        node.row = order.length;
        order.push(node);
        const kids = children.get(node.id) || [];
        // Reversed, because a stack hands them back the other way round.
        for (let i = kids.length - 1; i >= 0; i--) {
          stack.push([kids[i], this.tint(colour, kids[i].id), depth + 1]);
        }
      }
    }

    return { nodes, order, roots, children, firstIteration, lastIteration };
  },

  // ---- colour -----------------------------------------------------------

  /** A stable number in [0, 1) from an id and a channel. */
  _hash(id, channel) {
    let x = (id * 2654435761 + channel * 40503) >>> 0;
    x ^= x >>> 15;
    x = Math.imul(x, 2246822519) >>> 0;
    x ^= x >>> 13;
    return (x >>> 0) / 4294967296;
  },

  /** A founder gets its own colour, and nothing to inherit it from. */
  founderColour(id) {
    return { h: this._hash(id, 1) * 360, s: 62 + this._hash(id, 2) * 18, l: 58 };
  },

  /**
   * A mutation shifts its parent's colour a little.
   *
   * Small, so a family stays recognisable as a family over many generations,
   * but not so small that a long lineage never drifts. Saturation and
   * lightness are held inside a band, or deep lineages wander off to grey or
   * to white and stop being colours at all.
   */
  tint(parent, id) {
    const wrap = (v) => ((v % 360) + 360) % 360;
    const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
    return {
      h: wrap(parent.h + (this._hash(id, 3) - 0.5) * 16),
      s: clamp(parent.s + (this._hash(id, 4) - 0.5) * 12, 42, 88),
      l: clamp(parent.l + (this._hash(id, 5) - 0.5) * 12, 40, 74)
    };
  },

  css(colour, alpha = 1) {
    return `hsla(${colour.h.toFixed(1)}, ${colour.s.toFixed(0)}%, `
         + `${colour.l.toFixed(0)}%, ${alpha})`;
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

  /** The genotypes worth drawing, and where each one sits. */
  layout() {
    const f = this.forest;
    if (!f || !this.w) return null;

    const minLife = Number(this.minLifeEl.value) || 1;
    const shown = f.order.filter(n => n.span >= minLife);
    if (!shown.length) return null;

    const pad = { left: 8, right: 8, top: 10, bottom: 22 };
    const width = this.w - pad.left - pad.right;
    const height = this.h - pad.top - pad.bottom;
    const span = Math.max(1, f.lastIteration - f.firstIteration);
    const x = (t) => pad.left + ((t - f.firstIteration) / span) * width;
    const rows = shown.length;
    const y = (row) => pad.top + ((row + 0.5) / rows) * height;

    // Rows are renumbered over what is actually shown, so a filter closes the
    // gaps rather than leaving the picture full of holes.
    const row = new Map(shown.map((n, i) => [n.id, i]));
    const thickness = Math.max(0.6, Math.min(3.2, height / rows * 0.8));
    return { shown, x, y, row, pad, width, height, thickness, rows };
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
      ctx.fillText(this.forest ? 'Nothing lasted that long.' : 'No run loaded.',
                   this.w / 2, this.h / 2);
      return;
    }

    const f = this.forest;
    const { x, y, row, thickness } = plan;

    // The joins first, underneath, so a dense band reads as lines rather than
    // as a mesh.
    ctx.lineWidth = 0.7;
    for (const node of plan.shown) {
      const parent = f.nodes.get(node.parent);
      if (!parent || !row.has(parent.id)) continue;
      ctx.strokeStyle = this.css(node.colour, 0.38);
      ctx.beginPath();
      ctx.moveTo(x(node.born), y(row.get(parent.id)));
      ctx.lineTo(x(node.born), y(row.get(node.id)));
      ctx.stroke();
    }

    // Then each genotype, from the frame it appeared in to the last one it was
    // seen in, thickening with how many agents were carrying it.
    ctx.lineCap = 'round';
    for (const node of plan.shown) {
      const at = y(row.get(node.id));
      const wide = thickness * (1 + Math.min(2.2, Math.log2(1 + node.peak)));
      ctx.strokeStyle = this.css(node.colour, node === this.hovered ? 1 : 0.92);
      ctx.lineWidth = node === this.hovered ? wide + 2 : wide;
      ctx.beginPath();
      ctx.moveTo(x(node.born), at);
      ctx.lineTo(Math.max(x(node.died), x(node.born) + 1), at);
      ctx.stroke();
    }

    // The time axis.
    ctx.strokeStyle = 'rgba(190, 200, 215, 0.18)';
    ctx.lineWidth = 1;
    ctx.fillStyle = '#6b7c8d';
    ctx.font = '10px ui-monospace, monospace';
    ctx.textAlign = 'center';
    const ticks = 6;
    for (let i = 0; i <= ticks; i++) {
      const t = f.firstIteration + (f.lastIteration - f.firstIteration) * (i / ticks);
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
    const mx = event.clientX - box.left;

    let best = null, bestGap = Infinity;
    for (const node of plan.shown) {
      const at = plan.y(plan.row.get(node.id));
      const gap = Math.abs(at - my);
      if (gap < bestGap && mx >= plan.x(node.born) - 2 && mx <= plan.x(node.died) + 2) {
        best = node;
        bestGap = gap;
      }
    }
    const found = bestGap < 6 ? best : null;
    if (found === this.hovered) return;
    this.hovered = found;
    this.readoutEl.textContent = found
      ? `genotype ${found.id} — from iteration ${found.born} to ${found.died}, `
        + `held by up to ${formatNumber(found.peak)} agents, `
        + `${found.depth} mutation${found.depth === 1 ? '' : 's'} from its founder`
      : '';
    this.draw();
  }
};
