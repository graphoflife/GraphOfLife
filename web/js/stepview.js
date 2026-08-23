/*
 * One stage of the algorithm, drawn, and eased into from the last one.
 *
 * A stage is a graph — ids, tokens, edges — plus a bag of marks naming who did
 * what: who was born, who handed a link over, who staked what, who took a node,
 * which links carried nothing, who was removed. That is the shape the engine's
 * on_step hook hands out and the shape the Viewer's decision records already
 * hold, so nothing here is tied to the page it was written for.
 *
 * Nothing ever snaps. Everything drawn has a value it is heading towards and
 * moves a fraction of the way there each frame: where a node is, how solid it
 * is, how big it is, and where the camera is looking. Adopting a stage only
 * changes those targets, so stepping through the algorithm reads as one
 * continuous playback rather than a slideshow — which is the whole point, since
 * the thing being explained is a process.
 */
const StepView = {
  ink: {
    bg: '#0d1117',
    edge: 'rgba(190, 200, 215, 0.34)',
    node: '#8fa8e8',
    rich: '#f0a878',
    pale: '#e8eef8',
    good: '#7fd4a0',
    warn: '#e8896b',
    eye: '#101720',
    white: '#f2f6fb'
  },

  // How fast a value closes the gap to its target, per second. Low enough to
  // read as motion, high enough not to feel like lag.
  EASE: { position: 6.5, alpha: 4.5, camera: 3.2 },

  // How long a step holds its opening picture before the thing it is about
  // happens: births appear, the dead depart. Long enough to see what changed.
  LINGER: 2.0,

  create(canvas) {
    const view = {
      canvas,
      ctx: canvas.getContext('2d'),
      layout: new ForceLayout(),
      stage: null,
      effects: new Set(),
      // What is actually on screen, as opposed to where the layout says things
      // are. Everything in here chases the layout rather than jumping to it.
      shown: new Map(),          // id -> {x, y, alpha, r}
      camera: null,              // {scale, x, y}, eased
      since: 0,                  // seconds since this stage was adopted
      motes: []
    };
    Object.assign(view.layout, {
      charge: 46, linkStrength: 0.11, linkDistance: 34,
      centerStrength: 0.02, angularStrength: 0.2, damping: 0.84, theta: 1.2
    });
    view.layout.dimensions = 2;
    return view;
  },

  /**
   * Adopt a stage. Only targets change; nothing moves this instant.
   *
   * `effect` is a space-separated list of what the step wants drawn over the
   * graph — brains, eyes, the token arrival, envelopes, stakes, a conquest — so
   * a step can ask for several without the drawing code knowing which step it
   * is. A list rather than one name because these genuinely combine: the step
   * introducing the network wants brains in every node *and* the tokens
   * arriving.
   */
  show(view, stage, { effect = null, settle = 0 } = {}) {
    view.stage = stage;
    view.effects = new Set(String(effect || '').split(/\s+/).filter(Boolean));
    view.since = 0;

    const parents = new Array(stage.ids.length).fill(-1);
    const at = new Map(stage.ids.map((id, i) => [id, i]));
    for (const [parent, child] of (stage.marks.parents || [])) {
      const slot = at.get(child);
      if (slot !== undefined) parents[slot] = parent;
    }
    view.layout.setFrame(stage.ids, stage.edges, parents, view.shown.size > 0);
    view.layout.reheat(view.shown.size ? 0.35 : 1);
    for (let i = 0; i < settle; i++) {
      if (!view.layout.tick()) break;
    }
  },

  /** Advance the layout, then let everything drawn move towards it. */
  tick(view, dt) {
    if (!view.layout || !view.stage) return;
    view.since += dt;
    for (let i = 0; i < 2; i++) view.layout.tick();

    const stage = view.stage;
    const present = new Set(stage.ids);

    // The removed are drawn crossed out for a moment before they leave, so the
    // rule can be read before its result. They are in the stage — cleanup
    // snapshots the world it acted on — so taking them out is a matter of
    // dropping them from `present` once the pause is over.
    const dying = stage.marks && stage.marks.removed;
    if (dying && dying.length && view.since > this.LINGER) {
      for (const id of dying) present.delete(id);
    }

    // Newborns arrive a beat after the graph they are born into, so the step
    // starts on the picture the previous one ended on.
    const waiting = stage.step === 'repro.born' && view.since < this.LINGER
      ? new Set(stage.marks.born || []) : null;
    const most = Math.max(1, ...stage.tokens);
    const size = new Map(stage.ids.map((id, i) =>
      [id, 9 + 15 * Math.sqrt(stage.tokens[i] / most)]));

    const close = (was, want, rate) => was + (want - was) * Math.min(1, rate * dt);

    for (const id of stage.ids) {
      if (waiting && waiting.has(id)) continue;
      if (!present.has(id)) continue;
      const target = view.layout.pos.get(id);
      if (!target) continue;
      let node = view.shown.get(id);
      if (!node) {
        // Somewhere sensible to arrive from: beside whoever it came from, or
        // where it already is if the layout has an opinion.
        node = { x: target.x, y: target.y, alpha: 0, r: 1 };
        view.shown.set(id, node);
      }
      node.x = close(node.x, target.x, this.EASE.position);
      node.y = close(node.y, target.y, this.EASE.position);
      node.alpha = close(node.alpha, 1, this.EASE.alpha);
      node.r = close(node.r, size.get(id), this.EASE.position);
    }

    // Anyone no longer in the stage fades out where they stood, then goes.
    for (const [id, node] of view.shown) {
      if (present.has(id)) continue;
      node.alpha = close(node.alpha, 0, this.EASE.alpha);
      if (node.alpha < 0.02) view.shown.delete(id);
    }
  },

  // ---- drawing ----------------------------------------------------------

  draw(view, time, dt = 1 / 60) {
    const { ctx, canvas } = view;
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
    if (!view.stage || !view.shown.size) return;

    const place = this._camera(view, w, h, dt);
    const role = this._roles(view.stage);

    this._edges(ctx, view, place, role);
    this._nodes(ctx, view, place, role, time);
    this._effect(ctx, view, place, role, time, w, h);
  },

  /**
   * The camera, eased.
   *
   * Framing is recomputed from what is on screen every frame and then chased
   * rather than jumped to, so a stage that adds twenty agents widens the view
   * instead of flicking to a new one.
   */
  _camera(view, w, h, dt) {
    let loX = Infinity, hiX = -Infinity, loY = Infinity, hiY = -Infinity;
    for (const node of view.shown.values()) {
      if (node.alpha < 0.05) continue;
      loX = Math.min(loX, node.x); hiX = Math.max(hiX, node.x);
      loY = Math.min(loY, node.y); hiY = Math.max(hiY, node.y);
    }
    if (!Number.isFinite(loX)) return () => null;

    const pad = 54;
    // Spans are floored at something real and the scale is capped, because a
    // single agent has no extent at all: dividing the canvas by its width gave
    // a magnification of thousands, which put the one thing on screen somewhere
    // off it, and left the camera so far out of position that easing back for
    // the next step took seconds.
    const want = {
      scale: Math.min((w - pad * 2) / Math.max(40, hiX - loX),
                      (h - pad * 2) / Math.max(40, hiY - loY),
                      4),
      x: (loX + hiX) / 2,
      y: (loY + hiY) / 2
    };
    if (!view.camera) view.camera = { ...want };
    const k = Math.min(1, this.EASE.camera * dt);
    view.camera.scale += (want.scale - view.camera.scale) * k;
    view.camera.x += (want.x - view.camera.x) * k;
    view.camera.y += (want.y - view.camera.y) * k;

    const cam = view.camera;
    return (id) => {
      const node = view.shown.get(id);
      if (!node) return null;
      return { x: w / 2 + (node.x - cam.x) * cam.scale,
               y: h / 2 + (node.y - cam.y) * cam.scale,
               r: Math.max(3, node.r * Math.min(1, cam.scale * 0.9)),
               alpha: node.alpha };
    };
  },

  _roles(stage) {
    const marks = stage.marks || {};
    const key = (a, b) => (a < b ? `${a}|${b}` : `${b}|${a}`);
    return {
      step: stage.step,
      born: new Set(marks.born || []),
      parent: new Set((marks.parents || []).map(p => p[0])),
      handed: new Set((marks.handed || []).map(h => key(h[0], h[1]))),
      childLink: new Set((marks.parents || []).map(p => key(p[0], p[1]))),
      removed: new Set(marks.removed || []),
      cut: new Set((marks.cut || []).map(e => key(e[0], e[1]))),
      taken: new Set((marks.taken || []).map(t => t[0])),
      winner: new Set((marks.taken || []).map(t => t[1])),
      pairs: marks.taken || [],
      flow: new Map((marks.flow || []).map(f => [key(f[0], f[1]), f[2]])),
      // Who sent what to whom, which the undirected total cannot say. Both
      // ends of a link stake on each other, and showing only the sum made it
      // look like tokens travel one way.
      sent: (marks.staked || []).reduce((m, [to, from, amount]) => {
        if (from !== to) m.set(`${from}>${to}`, amount);
        return m;
      }, new Map()),
      key
    };
  },

  _edges(ctx, view, place, role) {
    for (const [a, b] of view.stage.edges) {
      const p = place(a), q = place(b);
      if (!p || !q) continue;
      const k = role.key(a, b);
      let colour = this.ink.edge, width = 2;

      let alpha = Math.min(p.alpha, q.alpha);

      if (role.handed.has(k) || role.childLink.has(k)) {
        colour = this.ink.rich;
        width = 3;
      } else if (role.flow.size) {
        // Where the tokens went. A link that carried some turns green and
        // thickens with the amount; one that carried none fades out here and
        // now, rather than waiting for the step the code prunes it in — what
        // matters to a reader is that the two are the same event.
        const carried = role.flow.get(k) || 0;
        if (carried > 0) {
          colour = 'rgba(127, 212, 160, 0.85)';
          width = 2 + Math.min(3, Math.log2(1 + carried));
        } else {
          colour = this.ink.warn;
          alpha *= Math.max(0, 1 - Math.max(0, view.since - 1.4) / 1.1);
        }
      } else if (role.cut.has(k)) {
        colour = this.ink.warn;
        width = 2.4;
      }

      ctx.globalAlpha = alpha;
      ctx.strokeStyle = colour;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.lineTo(q.x, q.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  },

  _nodes(ctx, view, place, role, time) {
    view.stage.ids.forEach((id, i) => {
      const p = place(id);
      if (!p) return;

      let colour = this.ink.node;
      if (role.born.has(id)) colour = this.ink.good;
      else if (role.taken.has(id)) colour = this.ink.rich;
      if (view._mutating === id) colour = '#f2cd5c';

      ctx.globalAlpha = p.alpha;
      if (role.winner.has(id) || role.parent.has(id)) {
        ctx.globalAlpha = p.alpha * 0.22;
        ctx.fillStyle = colour;
        ctx.beginPath(); ctx.arc(p.x, p.y, p.r * 2.1, 0, Math.PI * 2); ctx.fill();
        ctx.globalAlpha = p.alpha;
      }
      ctx.save();
      ctx.shadowColor = colour;
      ctx.shadowBlur = p.r * 0.9;
      ctx.fillStyle = colour;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      ctx.restore();

      if (role.removed.has(id)) {
        this._cross(ctx, p);
      } else if (view.effects.has('brains')) {
        this._brain(ctx, p, time, i, view._mutating === id ? view._mutatingUnit : -1);
      } else if (view.effects.has('eyes')) {
        this._eye(ctx, p, time, i, view, id);
      }
      ctx.globalAlpha = 1;
    });

    // Anyone on their way out, still fading where they stood.
    for (const [id, node] of view.shown) {
      if (view.stage.ids.includes(id)) continue;
      const p = place(id);
      if (!p) continue;
      ctx.globalAlpha = p.alpha;
      ctx.fillStyle = this.ink.warn;
      ctx.beginPath(); ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2); ctx.fill();
      ctx.globalAlpha = 1;
    }
  },

  /**
   * An eye: a white ball with a dark pupil, and nothing else.
   *
   * No outline and no lens. At this size an outline is most of what you see,
   * and what should be read is where the pupil is pointing.
   */
  _eye(ctx, p, time, seed, view, id) {
    // Scanning, not staring: they blink often and the gaze moves briskly, so a
    // field of them reads as a neighbourhood being read rather than a crowd
    // looking at nothing.
    const period = 1.15 + (seed % 9) * 0.13;
    const phase = (time + seed * 1.37) % period;
    const open = phase > 0.17 ? 1 : Math.abs(Math.cos((Math.PI * phase) / 0.17));
    const ball = p.r * 0.62;

    // Where it is looking: a neighbour, changing every so often, or itself.
    const look = view._gaze && view._gaze.get(id);
    let gx = 0, gy = 0;
    if (look) {
      const d = Math.hypot(look.x - p.x, look.y - p.y);
      if (d > 1) { gx = (look.x - p.x) / d; gy = (look.y - p.y) / d; }
    }

    ctx.fillStyle = this.ink.white;
    ctx.beginPath();
    ctx.ellipse(p.x, p.y, ball, ball * Math.max(0.06, open), 0, 0, Math.PI * 2);
    ctx.fill();
    if (open < 0.2) return;

    const pupil = ball * 0.46;
    ctx.fillStyle = this.ink.eye;
    ctx.beginPath();
    ctx.ellipse(p.x + gx * ball * 0.36, p.y + gy * ball * 0.36,
                pupil, pupil * open, 0, 0, Math.PI * 2);
    ctx.fill();
  },

  /**
   * The network inside an agent, drawn nearly to the edge of it.
   *
   * One size everywhere it appears, so it reads as the same object from the
   * opening step to the last. `hot` lights one unit red, which is how mutation
   * is shown.
   */
  _brain(ctx, p, time, seed, hot = -1) {
    const cols = [[-1, 2], [0, 3], [1, 2]];
    ctx.save();
    ctx.translate(p.x, p.y);
    ctx.strokeStyle = 'rgba(10,14,20,0.45)';
    ctx.lineWidth = Math.max(0.6, p.r * 0.045);
    const layers = [];
    cols.forEach(([cx, count], ci) => {
      const layer = [];
      for (let i = 0; i < count; i++) {
        layer.push({ x: cx * p.r * 0.62, y: (i - (count - 1) / 2) * p.r * 0.52 });
      }
      layers.push(layer);
      if (ci > 0) {
        for (const a of layers[ci - 1]) for (const b of layer) {
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
        }
      }
    });
    layers.flat().forEach((q, i) => {
      const lit = 0.5 + 0.5 * Math.max(0, Math.sin(time * 2.6 + seed - i * 0.5));
      ctx.fillStyle = (i === hot) ? '#ff5b52' : `rgba(10,14,20,${lit})`;
      ctx.beginPath(); ctx.arc(q.x, q.y, p.r * (i === hot ? 0.19 : 0.145), 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
  },

  _cross(ctx, p) {
    ctx.strokeStyle = this.ink.warn;
    ctx.lineWidth = Math.max(1.4, p.r * 0.24);
    const s = p.r * 0.5;
    ctx.beginPath();
    ctx.moveTo(p.x - s, p.y - s); ctx.lineTo(p.x + s, p.y + s);
    ctx.moveTo(p.x + s, p.y - s); ctx.lineTo(p.x - s, p.y + s);
    ctx.stroke();
  },

  // ---- the per-step animations -------------------------------------------

  _effect(ctx, view, place, role, time, w, h) {
    const along = (time * 0.5) % 1;

    // Tokens are drawn on every step, not only the one that introduces them.
    // They are the conserved thing the whole algorithm is about, so they should
    // never be off screen; on the opening steps they fly in and settle into the
    // same orbit they hold from then on.
    this._tokens(ctx, view, place, time, w, h);

    if (view.effects.has('messages')) {
      for (const [a, b] of view.stage.edges) {
        const p = place(a), q = place(b);
        if (!p || !q) continue;
        ctx.globalAlpha = Math.min(p.alpha, q.alpha);
        this._envelope(ctx, p.x + (q.x - p.x) * along, p.y + (q.y - p.y) * along, 5);
        this._envelope(ctx, q.x + (p.x - q.x) * along, q.y + (p.y - q.y) * along, 5, 0.6);
      }
      ctx.globalAlpha = 1;
      return;
    }
    if (view.effects.has('stakes')) {
      // These dots are the tokens themselves in transit, so they are the same
      // green they are everywhere else, and each direction of a link is drawn
      // separately — both ends stake on each other.
      ctx.save();
      ctx.fillStyle = this.ink.good;
      ctx.shadowColor = 'rgba(127, 212, 160, 0.9)';
      ctx.shadowBlur = 6;
      for (const [a, b] of view.stage.edges) {
        const p = place(a), q = place(b);
        if (!p || !q) continue;
        const alpha = Math.min(p.alpha, q.alpha);
        for (const [from, to, at, amount] of [
          [p, q, along, role.sent.get(`${a}>${b}`) || 0],
          [q, p, (along + 0.5) % 1, role.sent.get(`${b}>${a}`) || 0]
        ]) {
          if (!amount) continue;
          const dots = Math.min(8, amount);
          // Offset to one side, so the two directions do not sit on each other.
          const dx = to.x - from.x, dy = to.y - from.y;
          const len = Math.hypot(dx, dy) || 1;
          const ox = (-dy / len) * 4, oy = (dx / len) * 4;
          ctx.globalAlpha = alpha;
          for (let k = 0; k < dots; k++) {
            const t = (at + k / dots) % 1;
            ctx.beginPath();
            ctx.arc(from.x + dx * t + ox, from.y + dy * t + oy, 3, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }
      ctx.restore();
      ctx.globalAlpha = 1;
      return;
    }
    if (view.effects.has('conquer')) {
      // The winner's brain travelling into the node it took.
      const trip = Math.min(1, view.since / 2.2);
      for (const [node, winner] of role.pairs) {
        const from = place(winner), to = place(node);
        if (!from || !to) continue;
        const x = from.x + (to.x - from.x) * trip;
        const y = from.y + (to.y - from.y) * trip - Math.sin(trip * Math.PI) * 14;
        ctx.globalAlpha = Math.min(from.alpha, to.alpha) * (trip < 1 ? 1 : 0.4);
        this._brain(ctx, { x, y, r: Math.max(6, to.r * 0.7) }, time, node);
      }
      ctx.globalAlpha = 1;
    }
  },

  /**
   * Tokens, held in orbit around whoever holds them.
   *
   * One dot is a share of that agent's pile rather than one token, since a rich
   * agent can hold dozens; the count is capped so a node stays a node and does
   * not become a ring of dots.
   *
   * On the opening steps they arrive instead: the supply comes from nowhere in
   * the world, so it comes from outside the picture, each dot on its own
   * heading, and settles into exactly the orbit it will keep for every step
   * afterwards.
   */
  _tokens(ctx, view, place, time, w, h) {
    // Nothing is in orbit during the staking: every token is on a link.
    if (view.effects.has('stakes')) return;
    const stage = view.stage;
    const most = Math.max(1, ...stage.tokens);
    // Only the step that introduces them flies them in. The step after holds
    // the same orbit rather than replaying the arrival, so the dots simply stay
    // where they are while everything else moves on.
    const arriving = view.effects.has('arrive');
    const settled = arriving
      ? Math.min(1, Math.max(0, (view.since - 0.25) / 2.6))
      : 1;
    const eased = settled * settled * (3 - 2 * settled);
    const away = Math.max(w, h) * 1.15;

    ctx.save();
    ctx.fillStyle = this.ink.good;
    ctx.shadowColor = 'rgba(127, 212, 160, 0.85)';
    ctx.shadowBlur = 7;

    stage.ids.forEach((id, i) => {
      const p = place(id);
      if (!p) return;
      const dots = Math.max(1, Math.min(8, Math.round((stage.tokens[i] / most) * 7)));
      for (let k = 0; k < dots; k++) {
        const seed = (i * 7 + k * 13) % 97;
        const angle = (seed / 97) * Math.PI * 2 + time * 0.45;
        const orbit = p.r * 1.5;
        const toX = p.x + Math.cos(angle) * orbit;
        const toY = p.y + Math.sin(angle) * orbit;

        let x = toX, y = toY;
        if (arriving && eased < 1) {
          const fromX = p.x + Math.cos(seed) * away;
          const fromY = p.y + Math.sin(seed * 1.7) * away;
          x = fromX + (toX - fromX) * eased;
          y = fromY + (toY - fromY) * eased;
        }
        ctx.globalAlpha = p.alpha * (arriving ? Math.min(1, eased * 3) : 0.9);
        ctx.beginPath();
        ctx.arc(x, y, 3.1, 0, Math.PI * 2);
        ctx.fill();
      }
    });
    ctx.restore();
    ctx.globalAlpha = 1;
  },

  _envelope(ctx, x, y, s, alpha = 1) {
    ctx.save();
    ctx.globalAlpha *= alpha;
    ctx.fillStyle = '#eef2f8';
    ctx.strokeStyle = 'rgba(20,26,34,0.85)';
    ctx.lineWidth = 0.8;
    ctx.beginPath();
    ctx.rect(x - s, y - s * 0.66, s * 2, s * 1.32);
    ctx.fill(); ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x - s, y - s * 0.66);
    ctx.lineTo(x, y + s * 0.14);
    ctx.lineTo(x + s, y - s * 0.66);
    ctx.stroke();
    ctx.restore();
  },

  /**
   * Whose turn it is to be mutated, one after another.
   *
   * Every brain is jittered in the same instant; drawn that way it is a single
   * flicker across the whole graph and reads as nothing. Taken one agent at a
   * time it reads as what it is.
   */
  mutating(view, seconds) {
    if (!view.stage || !view.effects.has('mutate')) { view._mutating = null; return; }
    const ids = view.stage.ids;
    if (!ids.length) return;
    const at = Math.floor(seconds / 0.28) % ids.length;
    view._mutating = ids[at];
    view._mutatingUnit = Math.floor(seconds / 0.28 + at) % 7;
  },

  /**
   * Who each agent is looking at, changing every so often.
   *
   * Worked out once per stage rather than per frame, so a gaze holds still
   * long enough to be followed instead of flickering between neighbours.
   */
  gaze(view, seconds) {
    if (!view.stage) return;
    const adj = new Map(view.stage.ids.map(id => [id, []]));
    for (const [a, b] of view.stage.edges) {
      if (adj.has(a)) adj.get(a).push(b);
      if (adj.has(b)) adj.get(b).push(a);
    }
    view._gaze = new Map();
    const turn = Math.floor(seconds / 0.62);
    for (const id of view.stage.ids) {
      const near = adj.get(id) || [];
      const targets = [...near, id];                 // itself, last
      const pick = targets[Math.abs(turn + id) % targets.length];
      const node = view.shown.get(pick);
      if (node) view._gaze.set(id, node);
    }
  }
};
