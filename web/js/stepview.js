/*
 * One stage of the algorithm, drawn.
 *
 * A stage is a graph — ids, tokens, edges — plus a bag of marks naming who did
 * what: who was born, who handed an edge over, who staked what where, who took
 * a node, which links carried nothing, who was removed. That is the shape the
 * engine's on_step hook hands out, and it is also the shape the Viewer's own
 * decision records already hold, so nothing here is tied to the page it was
 * written for. Give it stages from anywhere and it will draw them.
 *
 * It keeps its own force layout so positions carry from one stage to the next.
 * That is the whole reason this is worth having: the same agents stay in the
 * same places while the rules act on them, which is what makes it possible to
 * watch one agent be born, stake, lose and be removed.
 */
const StepView = {
  ink: {
    bg: '#0d1117',
    edge: 'rgba(190, 200, 215, 0.30)',
    node: '#8fa8e8',
    rich: '#f0a878',
    pale: '#e8eef8',
    good: '#7fd4a0',
    warn: '#e8896b',
    dead: 'rgba(190, 200, 215, 0.16)',
    eye: '#05070a',
    text: '#8b9bab'
  },

  /**
   * A view over one canvas.
   *
   * `settle` is how many layout ticks to run when a stage is adopted. Enough
   * that the graph has arranged itself before it is looked at, since nobody is
   * watching it converge — the reader is reading.
   */
  create(canvas, { settle = 220 } = {}) {
    const view = {
      canvas,
      ctx: canvas.getContext('2d'),
      layout: new ForceLayout(),
      stage: null,
      blinkFrom: Math.random() * 10,
      settleTicks: settle
    };
    Object.assign(view.layout, {
      charge: 34, linkStrength: 0.12, linkDistance: 26,
      centerStrength: 0.02, angularStrength: 0.2, damping: 0.82, theta: 1.2
    });
    view.layout.dimensions = 2;
    return view;
  },

  /**
   * Adopt a stage, carrying positions for everyone who was already here.
   *
   * `parents` lets a newborn appear beside the agent it came from rather than
   * somewhere random, which matters more here than anywhere: the whole point
   * of the reproduction step is that you can see where the child came from.
   */
  show(view, stage, { carry = true } = {}) {
    view.stage = stage;
    const parents = new Array(stage.ids.length).fill(-1);
    const at = new Map(stage.ids.map((id, i) => [id, i]));
    for (const [parent, child] of (stage.marks.parents || [])) {
      const slot = at.get(child);
      if (slot !== undefined) parents[slot] = parent;
    }
    view.layout.setFrame(stage.ids, stage.edges, parents, carry);
    view.layout.reheat(carry ? 0.5 : 1);
    for (let i = 0; i < view.settleTicks; i++) {
      if (!view.layout.tick()) break;
    }
  },

  /** A few more ticks, so the picture keeps breathing while it is read. */
  tick(view) {
    if (view.layout) view.layout.tick();
  },

  // ---- drawing ----------------------------------------------------------

  draw(view, time = 0) {
    const { ctx, canvas, stage } = view;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const box = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.round(box.width));
    const h = Math.max(1, Math.round(box.height));
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = this.ink.bg;
    ctx.fillRect(0, 0, w, h);
    if (!stage) return;

    const place = this._fit(view, w, h);
    const marks = stage.marks || {};
    const role = this._roles(stage, marks);
    const tokens = new Map(stage.ids.map((id, i) => [id, stage.tokens[i]]));

    this._edges(ctx, view, place, stage, role, time);
    this._nodes(ctx, view, place, stage, role, tokens, time);
    this._motes(ctx, view, place, stage, role, time);
  },

  /**
   * Positions scaled into the canvas, with room for the biggest agent.
   *
   * Read from the layout's own map rather than its flat array. The array is
   * only filled by syncPositions(), which the worker-backed client calls and
   * nothing here does — reading it without that gave every agent the origin,
   * so the whole graph drew as a single dot in the middle.
   */
  _fit(view, w, h) {
    const pos = view.layout.pos;
    if (!pos || !pos.size) return () => null;

    let loX = Infinity, hiX = -Infinity, loY = Infinity, hiY = -Infinity;
    for (const p of pos.values()) {
      loX = Math.min(loX, p.x); hiX = Math.max(hiX, p.x);
      loY = Math.min(loY, p.y); hiY = Math.max(hiY, p.y);
    }
    const pad = 34;
    const spanX = Math.max(1e-3, hiX - loX);
    const spanY = Math.max(1e-3, hiY - loY);
    const scale = Math.min((w - pad * 2) / spanX, (h - pad * 2) / spanY);
    const midX = (loX + hiX) / 2, midY = (loY + hiY) / 2;
    return (id) => {
      const p = pos.get(id);
      if (!p) return null;
      return { x: w / 2 + (p.x - midX) * scale, y: h / 2 + (p.y - midY) * scale };
    };
  },

  /** Who is what, this stage. */
  _roles(stage, marks) {
    const key = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
    return {
      born: new Set(marks.born || []),
      parent: new Set((marks.parents || []).map(p => p[0])),
      handed: new Set((marks.handed || []).map(h => key(h[0], h[1]))),
      removed: new Set(marks.removed || []),
      cut: new Set((marks.cut || []).map(e => key(e[0], e[1]))),
      taken: new Set((marks.taken || []).map(t => t[0])),
      winner: new Set((marks.taken || []).map(t => t[1])),
      revolted: new Set((marks.revolts || []).filter(r => r[1] > 0).map(r => r[0])),
      flow: new Map((marks.flow || []).map(f => [key(f[0], f[1]), f[2]])),
      key
    };
  },

  _edges(ctx, view, place, stage, role, time) {
    for (const [a, b] of stage.edges) {
      const p = place(a), q = place(b);
      if (!p || !q) continue;
      const k = role.key(a, b);
      let colour = this.ink.edge, width = 1.1;

      if (role.cut.has(k)) { colour = this.ink.warn; width = 1.6; }
      else if (role.handed.has(k)) { colour = this.ink.rich; width = 1.8; }
      else if (role.flow.size) {
        const carried = role.flow.get(k) || 0;
        if (carried === 0) { colour = 'rgba(232,137,107,0.45)'; }
        else { colour = 'rgba(232,238,248,0.5)'; width = 1.4; }
      }
      if (role.removed.has(a) || role.removed.has(b)) colour = this.ink.dead;

      ctx.strokeStyle = colour;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(q.x, q.y);
      ctx.stroke();
    }
  },

  _nodes(ctx, view, place, stage, role, tokens, time) {
    const most = Math.max(1, ...stage.tokens);
    stage.ids.forEach((id, i) => {
      const p = place(id);
      if (!p) return;
      const held = tokens.get(id) || 0;
      const r = 5 + 7 * Math.sqrt(held / most);

      let colour = this.ink.node;
      if (role.removed.has(id)) colour = this.ink.dead;
      else if (role.born.has(id)) colour = this.ink.good;
      else if (role.taken.has(id)) colour = this.ink.rich;
      else if (role.revolted.has(id)) colour = this.ink.good;

      if (role.winner.has(id) || role.parent.has(id)) {
        ctx.globalAlpha = 0.2;
        ctx.fillStyle = colour;
        ctx.beginPath();
        ctx.arc(p.x, p.y, r * 2.3, 0, Math.PI * 2);
        ctx.fill();
        ctx.globalAlpha = 1;
      }

      ctx.fillStyle = colour;
      ctx.beginPath();
      ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
      ctx.fill();

      if (role.removed.has(id)) this._cross(ctx, p, r);
      else if (stage.step.endsWith('observe')) this._eye(ctx, p, r, time, i);
    });
  },

  /** An eye, the same one the mark uses: a lens, an iris, a round pupil. */
  _eye(ctx, p, r, time, seed) {
    const period = 3.1 + (seed % 11) * 0.31;
    const phase = (time + seed * 1.37) % period;
    const open = phase > 0.15 ? 1 : Math.abs(Math.cos((Math.PI * phase) / 0.15));
    const halfW = r * 0.74, halfH = r * 0.42;

    if (open < 0.14) {
      ctx.strokeStyle = this.ink.eye;
      ctx.lineWidth = Math.max(0.9, r * 0.14);
      ctx.beginPath();
      ctx.moveTo(p.x - halfW, p.y);
      ctx.lineTo(p.x + halfW, p.y);
      ctx.stroke();
      return;
    }
    const lens = () => {
      const lid = halfH * 2 * open;
      ctx.beginPath();
      ctx.moveTo(p.x - halfW, p.y);
      ctx.quadraticCurveTo(p.x, p.y - lid, p.x + halfW, p.y);
      ctx.quadraticCurveTo(p.x, p.y + lid, p.x - halfW, p.y);
      ctx.closePath();
    };
    lens();
    ctx.fillStyle = '#f4f7fb';
    ctx.fill();
    ctx.save();
    lens(); ctx.clip();
    const iris = Math.min(halfH * 1.1, r * 0.36);
    ctx.globalAlpha = 0.34;
    ctx.fillStyle = this.ink.eye;
    ctx.beginPath(); ctx.arc(p.x, p.y, iris, 0, Math.PI * 2); ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(p.x, p.y, iris * 0.52, 0, Math.PI * 2); ctx.fill();
    ctx.restore();
    lens();
    ctx.strokeStyle = this.ink.eye;
    ctx.lineWidth = Math.max(0.7, r * 0.11);
    ctx.stroke();
  },

  /** A dead agent's eyes, for the stage where it is removed. */
  _cross(ctx, p, r) {
    ctx.strokeStyle = this.ink.warn;
    ctx.lineWidth = Math.max(1, r * 0.28);
    const s = r * 0.55;
    ctx.beginPath();
    ctx.moveTo(p.x - s, p.y - s); ctx.lineTo(p.x + s, p.y + s);
    ctx.moveTo(p.x + s, p.y - s); ctx.lineTo(p.x - s, p.y + s);
    ctx.stroke();
  },

  /** Envelopes on the observation stages, tokens on the staking one. */
  _motes(ctx, view, place, stage, role, time) {
    const along = (time * 0.55) % 1;
    if (stage.step.endsWith('observe')) {
      for (const [a, b] of stage.edges) {
        const p = place(a), q = place(b);
        if (!p || !q) continue;
        this._envelope(ctx, p.x + (q.x - p.x) * along, p.y + (q.y - p.y) * along, 3.6);
        this._envelope(ctx, q.x + (p.x - q.x) * along, q.y + (p.y - q.y) * along, 3.6, 0.55);
      }
      return;
    }
    if (stage.step === 'game.stake') {
      ctx.fillStyle = this.ink.pale;
      for (const [a, b] of stage.edges) {
        const carried = role.flow.get(role.key(a, b)) || 0;
        if (!carried) continue;
        const p = place(a), q = place(b);
        if (!p || !q) continue;
        ctx.beginPath();
        ctx.arc(p.x + (q.x - p.x) * along, p.y + (q.y - p.y) * along, 2, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  },

  _envelope(ctx, x, y, s, alpha = 0.9) {
    ctx.save();
    ctx.globalAlpha = alpha;
    ctx.fillStyle = '#eef2f8';
    ctx.strokeStyle = 'rgba(20,26,34,0.85)';
    ctx.lineWidth = 0.7;
    ctx.beginPath();
    ctx.rect(x - s, y - s * 0.66, s * 2, s * 1.32);
    ctx.fill(); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - s, y - s * 0.66);
    ctx.lineTo(x, y + s * 0.14);
    ctx.lineTo(x + s, y - s * 0.66);
    ctx.stroke();
    ctx.restore();
  }
};
