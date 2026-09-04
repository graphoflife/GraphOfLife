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
    eye: '#05070a',
    iris: '#5ab6ef',
    lost: '#ef5f52',
    white: '#f2f6fb'
  },

  // How fast a value closes the gap to its target, per second. Low enough to
  // read as motion, high enough not to feel like lag.
  EASE: { position: 6.5, alpha: 4.5, camera: 3.2, pupil: 7.5 },

  // How long an eye holds one neighbour before moving to the next. The scan
  // cone is sized to arrive exactly at the end of it.
  GAZE: 0.9,

  // How many dots the richest agent in the run is drawn with.
  TOKEN_DOTS: 15,

  // The game, in the order it is read: the piles as they stand, then the
  // staking, then the brains of whoever won. Each waits for the last.
  // `spread` is how much of the crossing the dots are strung out over: they
  // leave across the first slice of it and land across the last.
  GAME: { hold: 1.0, travel: 2.2, spread: 0.24 },

  // A birth: how long the child stands there before what it was given starts
  // crossing the link, and how long the crossing takes.
  BIRTH: { wait: 0.5, hold: 0.9, travel: 1.7, spread: 0.24 },

  // A conquest: how long a brain takes to travel, and how far apart the
  // journeys are started. Staggered because a hundred nodes changing colour on
  // the same frame reads as a scene change rather than as a hundred arrivals.
  CONQUEST: { travel: 1.7, stagger: 1.3, fade: 0.55 },

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
    // A cleanup kills twice, in order, and the two reasons are different
    // enough to be worth separating: first everyone holding nothing, then
    // everyone the starving cut adrift from the main body. The recording keeps
    // one list, but the snapshot it was taken over still has the tokens, and
    // starvation runs first — so anyone removed while holding nothing starved,
    // and anyone removed still holding something was stranded.
    const doomed = new Set(stage.marks.removed || []);
    view.waves = [[], []];
    stage.ids.forEach((id, i) => {
      if (doomed.has(id)) view.waves[stage.tokens[i] > 0 ? 1 : 0].push(id);
    });

    view._plan = null;

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
    // dropping them from `present` once their wave is over.
    if (view.waves) {
      view.waves.forEach((wave, w) => {
        if (view.since > this.LINGER * (w + 1)) for (const id of wave) present.delete(id);
      });
    }

    // A newborn arrives a beat after the graph it is born into, so the step
    // opens on the picture the last one closed on and the birth is something
    // that happens rather than something already there.
    const waiting = stage.step === 'repro.born' && view.since < this.BIRTH.wait
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
    this._scan(ctx, view, place);
    this._conquest(view, role);
    this._nodes(ctx, view, place, role, time, dt);
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
        colour = this.ink.good;
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

  _nodes(ctx, view, place, role, time, dt) {
    view.stage.ids.forEach((id, i) => {
      const p = place(id);
      if (!p) return;

      let colour = this.ink.node;
      if (role.born.has(id)) colour = this.ink.good;
      else if (view._conquest) {
        // Green the moment the step opens for anyone who gets to copy itself,
        // and red only once somebody else's brain has actually arrived. An
        // agent can be both: it wins a neighbour and loses its own node.
        if (role.winner.has(id)) colour = this.ink.good;
        const trip = view._conquest.get(id);
        if (trip && trip.t >= 1) colour = this.ink.lost;
      } else if (role.taken.has(id) && !view.effects.has('conquer')) {
        // Only where the outcome is not the thing being animated. On the game
        // step it is, and colouring the taken nodes while their tokens are
        // still crossing the graph answers the question the step is asking.
        colour = this.ink.rich;
      }
      if (view._mutated && view._mutated.has(id)) colour = '#f2cd5c';

      // Struck out, and coloured to match: the mark and the fact should not be
      // two separate things to notice.
      const wave = view.waves ? view.waves.findIndex(w => w.includes(id)) : -1;
      const struck = wave >= 0 && view.since > this.LINGER * wave;
      if (struck) colour = this.ink.lost;

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

      if (struck) {
        this._cross(ctx, p);
      } else if (view.effects.has('brains')) {
        this._brain(ctx, p, time, i,
                    view._mutated && view._mutated.has(id) ? view._mutated.get(id) : -1);
      } else if (view.effects.has('eyes')) {
        // Resolved here, where screen positions exist. Passing the layout's
        // own coordinates in meant every pupil was aimed at a point in a
        // different space — which put all of them in the same corner.
        const look = view._gaze && view._gaze.get(id);
        this._eye(ctx, view, p, time, id, look ? place(look.at) : null, dt);
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
   * A number in [0, 1) from an id. Only has to look unrelated to its
   * neighbours', which the golden ratio does about as well as anything.
   */
  _hash(id) {
    const x = (id + 1) * 0.6180339887498949;
    return x - Math.floor(x);
  },

  /**
   * An eye: a white ball, a light blue iris with spokes, a round black pupil.
   *
   * The same eye as the mark in the header and the tab, drawn in the same
   * proportions, so the thing in the corner of the page and the thing in the
   * picture are recognisably one object.
   *
   * Blinks are per-eye. They share one rhythm, but each starts wherever its own
   * id puts it, so a field of them never blinks in chorus.
   */
  _eye(ctx, view, p, time, id, look, dt = 1 / 60) {
    const period = 0.95 + this._hash(id * 7 + 3) * 0.85;
    const phase = (time + this._hash(id) * period) % period;
    const open = phase > 0.17 ? 1 : Math.abs(Math.cos((Math.PI * phase) / 0.17));
    const ball = p.r * 0.62;

    // Where it is looking: a neighbour, changing every so often, or itself —
    // and looking at itself means the iris sits in the middle.
    let gx = 0, gy = 0;
    if (look) {
      const d = Math.hypot(look.x - p.x, look.y - p.y);
      if (d > 1) { gx = (look.x - p.x) / d; gy = (look.y - p.y) / d; }
    }

    // Swung rather than snapped. The choice of who to look at changes on the
    // instant, and drawing it that way made every eye in the graph flick;
    // an eye moving to the next neighbour is the part worth watching.
    if (!view._pupil) view._pupil = new Map();
    let dir = view._pupil.get(id);
    if (!dir) view._pupil.set(id, dir = { x: gx, y: gy });
    const swing = Math.min(1, this.EASE.pupil * dt);
    dir.x += (gx - dir.x) * swing;
    dir.y += (gy - dir.y) * swing;
    gx = dir.x; gy = dir.y;

    ctx.fillStyle = this.ink.white;
    ctx.beginPath();
    ctx.ellipse(p.x, p.y, ball, ball * Math.max(0.06, open), 0, 0, Math.PI * 2);
    ctx.fill();
    if (open < 0.2) return;

    // The mark's proportions, against the ball rather than against the agent.
    const cx = p.x + gx * ball * 0.24;
    const cy = p.y + gy * ball * 0.24;
    const iris = ball * 0.60;
    const pupil = ball * 0.31;

    ctx.fillStyle = this.ink.iris;
    ctx.beginPath();
    ctx.ellipse(cx, cy, iris, iris * open, 0, 0, Math.PI * 2);
    ctx.fill();

    // Spokes, only once there are enough pixels for nine of them to be nine
    // rather than a smudge.
    if (iris > 3.6) {
      ctx.save();
      ctx.globalAlpha = ctx.globalAlpha * 0.30;
      ctx.strokeStyle = this.ink.eye;
      ctx.lineWidth = Math.max(0.5, ball * 0.075);
      ctx.beginPath();
      for (let k = 0; k < 9; k++) {
        const a = (k / 9) * Math.PI * 2 + 0.2;
        ctx.moveTo(cx + Math.cos(a) * pupil * 1.12, cy + Math.sin(a) * pupil * 1.12 * open);
        ctx.lineTo(cx + Math.cos(a) * iris * 0.94, cy + Math.sin(a) * iris * 0.94 * open);
      }
      ctx.stroke();
      ctx.restore();
    }

    ctx.fillStyle = this.ink.eye;
    ctx.beginPath();
    ctx.ellipse(cx, cy, pupil, pupil * open, 0, 0, Math.PI * 2);
    ctx.fill();
  },

  /**
   * The cone an eye sweeps while it is reading a neighbour.
   *
   * An eye pointing somewhere is a small thing to see. The cone is the reading
   * itself: it opens at the agent and runs out along the link, reaching the
   * neighbour just as the look ends and the next one begins. Looking at itself
   * has no direction to run along, so it gets a ring that widens instead.
   *
   * Drawn under the agents rather than over them, so nothing it sweeps across
   * is hidden by it.
   */
  _scan(ctx, view, place) {
    if (!view.effects.has('eyes') || !view._gaze) return;
    ctx.save();
    for (const [id, look] of view._gaze) {
      const p = place(id);
      if (!p) continue;

      // Fast at the start and easing into the arrival, which is what a sweep
      // looks like; then out of the way before the next one starts.
      const t = look.through;
      const reach = t * t * (3 - 2 * t);
      const fade = p.alpha * Math.min(1, (1 - t) * 3.4) * Math.min(1, t * 6);

      if (look.at === id) {
        // Kept close to the agent. Grown to the size the cones reach it read
        // as a bubble around the node rather than as the node reading itself,
        // and a graph of them was mostly circles.
        const r = p.r * (1.15 + 0.85 * reach);
        ctx.globalAlpha = fade * 0.75;
        ctx.strokeStyle = this.ink.iris;
        ctx.lineWidth = Math.max(1, p.r * 0.07);
        ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, Math.PI * 2); ctx.stroke();
        continue;
      }

      const q = place(look.at);
      if (!q) continue;
      const d = Math.hypot(q.x - p.x, q.y - p.y);
      if (d < 1) continue;
      const a = Math.atan2(q.y - p.y, q.x - p.x);
      const far = Math.max(p.r, d * reach);
      const half = 0.30;

      const grad = ctx.createLinearGradient(
        p.x, p.y, p.x + Math.cos(a) * far, p.y + Math.sin(a) * far);
      grad.addColorStop(0, 'rgba(90, 182, 239, 0)');
      grad.addColorStop(0.35, 'rgba(90, 182, 239, 0.40)');
      grad.addColorStop(1, 'rgba(90, 182, 239, 0.10)');
      ctx.globalAlpha = fade;
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.moveTo(p.x, p.y);
      ctx.arc(p.x, p.y, far, a - half, a + half);
      ctx.closePath();
      ctx.fill();

      // The front of the sweep, so the growth has an edge to follow.
      ctx.globalAlpha = fade * 0.8;
      ctx.strokeStyle = this.ink.iris;
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(p.x, p.y, far, a - half, a + half);
      ctx.stroke();
    }
    ctx.restore();
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

  // Three times the agent across, so it is a mark struck through the agent
  // rather than a detail inside it.
  _cross(ctx, p) {
    ctx.save();
    ctx.strokeStyle = this.ink.lost;
    ctx.lineWidth = Math.max(2.8, p.r * 0.36);
    ctx.lineCap = 'round';
    const s = p.r * 1.5;
    ctx.beginPath();
    ctx.moveTo(p.x - s, p.y - s); ctx.lineTo(p.x + s, p.y + s);
    ctx.moveTo(p.x + s, p.y - s); ctx.lineTo(p.x - s, p.y + s);
    ctx.stroke();
    ctx.restore();
  },

  // ---- the per-step animations -------------------------------------------

  _effect(ctx, view, place, role, time, w, h) {
    // Tokens are drawn on every step, not only the one that introduces them.
    // They are the conserved thing the whole algorithm is about, so they should
    // never be off screen; on the opening steps they fly in and settle into the
    // same orbit they hold from then on.
    this._tokens(ctx, view, place, time, w, h);

    if (view.effects.has('inherit')) {
      // What a child was given, crossing the link it was given it down. Only
      // one direction to draw, so the amount can sit on the link itself.
      const share = this._inheritance(view);
      if (share.at > 0 && share.at < 1) {
        ctx.save();
        ctx.fillStyle = this.ink.good;
        ctx.shadowColor = 'rgba(127, 212, 160, 0.9)';
        ctx.shadowBlur = 6;
        for (const [parent, child] of (view.stage.marks.parents || [])) {
          const from = place(parent), to = place(child);
          if (!from || !to) continue;
          const amount = share.given(child);
          if (amount <= 0) continue;
          const dots = Math.min(8, amount);
          const spread = this.BIRTH.spread;
          ctx.globalAlpha = Math.min(from.alpha, to.alpha);
          for (let k = 0; k < dots; k++) {
            const t = Math.max(0, Math.min(1,
              (share.at - (k / dots) * spread) / (1 - spread)));
            ctx.beginPath();
            ctx.arc(from.x + (to.x - from.x) * t, from.y + (to.y - from.y) * t,
                    3, 0, Math.PI * 2);
            ctx.fill();
          }
        }
        ctx.restore();
      }
    }

    if (view.effects.has('stakes')) {
      // These dots are the tokens themselves in transit, so they are the same
      // green they are everywhere else, and each direction of a link is drawn
      // separately — both ends stake on each other. They leave when the piles
      // start draining and land when the piles have finished filling, because
      // they are the same tokens.
      const moving = this._staking(view);
      if (moving.raw > 0 && moving.raw < 1.12) {
        ctx.save();
        ctx.fillStyle = this.ink.good;
        ctx.shadowColor = 'rgba(127, 212, 160, 0.9)';
        ctx.shadowBlur = 6;
        for (const [a, b] of view.stage.edges) {
          const p = place(a), q = place(b);
          if (!p || !q) continue;
          const alpha = Math.min(p.alpha, q.alpha);
          for (const [from, to, amount] of [
            [p, q, role.sent.get(`${a}>${b}`) || 0],
            [q, p, role.sent.get(`${b}>${a}`) || 0]
          ]) {
            if (!amount) continue;
            const dots = Math.min(8, amount);
            // Offset to one side, so the two directions do not sit on each other.
            const dx = to.x - from.x, dy = to.y - from.y;
            const len = Math.hypot(dx, dy) || 1;
            const ox = (-dy / len) * 4, oy = (dx / len) * 4;
            ctx.globalAlpha = alpha;
            for (let k = 0; k < dots; k++) {
              const spread = this.GAME.spread;
              const t = Math.max(0, Math.min(1,
                (moving.at - (k / dots) * spread) / (1 - spread)));
              ctx.beginPath();
              ctx.arc(from.x + dx * t + ox, from.y + dy * t + oy, 3, 0, Math.PI * 2);
              ctx.fill();
            }
          }
        }
        ctx.restore();
        ctx.globalAlpha = 1;
      }
    }

    if (view.effects.has('conquer') && view._conquest) {
      // The winner's brain travelling into the node it took, lit from behind
      // so it can be followed across a crowded graph. Without the backlight it
      // is one more small dark network among forty of them.
      for (const [node, trip] of view._conquest) {
        const from = place(trip.winner), to = place(node);
        if (!from || !to) continue;
        const t = trip.t;
        const x = from.x + (to.x - from.x) * t;
        const y = from.y + (to.y - from.y) * t - Math.sin(t * Math.PI) * 14;
        const r = Math.max(6, to.r * 0.7);

        // Gone shortly after it lands: from then on the node is drawing the
        // brain itself, and two copies of it in one place is just a smudge.
        const done = Math.max(0, 1 - trip.after / this.CONQUEST.fade);
        if (done <= 0) continue;
        const near = Math.min(from.alpha, to.alpha) * done;

        const pulse = 0.55 + 0.45 * Math.sin(time * 4.4 + this._hash(node) * Math.PI * 2);
        const halo = ctx.createRadialGradient(x, y, r * 0.5, x, y, r * 2.1);
        halo.addColorStop(0, `rgba(255, 255, 255, ${(0.55 + 0.4 * pulse).toFixed(3)})`);
        halo.addColorStop(0.55, `rgba(255, 255, 255, ${(0.22 * pulse).toFixed(3)})`);
        halo.addColorStop(1, 'rgba(255, 255, 255, 0)');
        ctx.globalAlpha = near;
        ctx.fillStyle = halo;
        ctx.beginPath(); ctx.arc(x, y, r * 2.1, 0, Math.PI * 2); ctx.fill();

        this._brain(ctx, { x, y, r }, time, node);
      }
      ctx.globalAlpha = 1;
    }
  },

  /**
   * What each newborn was handed, and how far it has got.
   *
   * A child starts with exactly what it was given, so the snapshot's own count
   * for it is the amount — nothing else has happened to it yet. The snapshot
   * is taken after the spending, though, so the parent is already short of it:
   * to show the handover the parent has to be given it back until the tokens
   * leave, or they appear out of nowhere and nobody is seen paying for them.
   *
   * Same schedule as the game's staking, for the same reason: a pile should
   * lose what it sends when the tokens set off and gain what it is sent when
   * they land.
   */
  _inheritance(view) {
    const stage = view.stage;
    const amount = new Map(stage.ids.map((id, i) => [id, stage.tokens[i]]));
    const out = new Map(), inc = new Map();
    for (const [parent, child] of (stage.marks.parents || [])) {
      const paid = amount.get(child) || 0;
      if (paid <= 0) continue;
      out.set(parent, (out.get(parent) || 0) + paid);
      inc.set(child, (inc.get(child) || 0) + paid);
    }
    const g = this.BIRTH;
    const run = (view.since - g.hold) / g.travel;
    const at = Math.max(0, Math.min(1, run));
    const eased = at * at * (3 - 2 * at);
    const ramp = (a, b) => Math.max(0, Math.min(1, (eased - a) / (b - a)));
    const gone = ramp(0, g.spread);
    const come = ramp(1 - g.spread, 1);
    return {
      given: id => inc.get(id) || 0,
      at: eased,
      raw: run,
      held: (id, had) =>
        had + (out.get(id) || 0) * (1 - gone) - (inc.get(id) || 0) * (1 - come)
    };
  },

  /**
   * How far each conquering brain has got, and whether it has landed.
   *
   * Kept apart from the drawing because the node colours depend on it: a node
   * turns over when the brain reaches it, not when the step opens. Showing the
   * result from the first frame gave away the answer before the journey that
   * decides it — and the winners, which are green from the first frame, are
   * exactly the ones the reader should be watching set out.
   */
  _conquest(view, role) {
    if (!view.effects.has('conquer')) { view._conquest = null; return; }
    // On the step that shows the staking too, nothing about the outcome exists
    // until the tokens have landed: that is what decides it.
    const opens = view.effects.has('stakes')
      ? this.GAME.hold + this.GAME.travel + 0.3 : 0;
    const since = view.since - opens;
    if (since < 0) { view._conquest = null; return; }

    const m = new Map();
    for (const [node, winner] of role.pairs) {
      const start = this._hash(node * 13 + 5) * this.CONQUEST.stagger;
      const run = (since - start) / this.CONQUEST.travel;
      m.set(node, {
        winner,
        t: Math.max(0, Math.min(1, run)),
        after: Math.max(0, since - start - this.CONQUEST.travel)
      });
    }
    view._conquest = m;
  },

  /**
   * Tokens, held in orbit around whoever holds them.
   *
   * One dot is a share of that agent's pile rather than one token, since a rich
   * agent can hold dozens. The count is the log of the pile against one fixed
   * reference for the whole run — the 95th percentile of the richest holding in
   * each recorded stage — so fifteen dots means the same amount of money in
   * every step. Measuring against the richest agent in the current stage
   * instead made the scale move under the reader: the same pile drew a
   * different number of dots depending on who else happened to be holding
   * something that step.
   *
   * On the opening steps they arrive instead: the supply comes from nowhere in
   * the world, so it comes from outside the picture, each dot on its own
   * heading, and settles into exactly the orbit it will keep for every step
   * afterwards.
   */
  // The opening supply: how long the ball takes to go round firing, and how
  // long one token spends in the air.
  // A short flight on purpose. Give a token a third of the sweep to reach its
  // agent and a third of every shot is in the air at once, which draws the
  // ring rather than the firing: it looked like a comet trailing the ball
  // instead of tokens being put into agents.
  SUPPLY: { sweep: 1.5, flight: 0.2, lead: 0.35 },

  /**
   * The token supply, fired in rather than drifting in from off screen.
   *
   * Every token in the world arrives in one pass: a ball circles the graph and
   * shoots them into the agents as it comes round to each one, emptying — and
   * shrinking — as it goes. Shots are ordered by the angle of the agent they
   * are going to, so the ball is always firing at whatever it is pointing at,
   * and the bunching that comes of agents not being spread evenly round the
   * circle is what makes it read as a burst rather than a metronome.
   */
  _supply(view, place, count, w, h) {
    const stage = view.stage;
    let cx = 0, cy = 0, n = 0, reach = 0;
    const at = [];
    stage.ids.forEach((id, i) => {
      const p = place(id);
      if (!p || stage.tokens[i] <= 0) return;
      cx += p.x; cy += p.y; n++;
      at.push({ id, p, dots: count(stage.tokens[i]) });
    });
    if (!n) return null;
    cx /= n; cy /= n;
    for (const a of at) reach = Math.max(reach, Math.hypot(a.p.x - cx, a.p.y - cy));
    // Outside the graph, but never outside the canvas: framed on the agents,
    // a ring drawn past the widest of them is a ring drawn off screen, and the
    // ball firing from beyond the edge is a ball nobody sees.
    const ring = Math.min(reach * 1.12 + 24, Math.min(w, h) * 0.46);

    // One entry per token, in the order the ball comes round to them — worked
    // out once and kept. Sorting live, on positions that are still settling
    // into the step, let two agents swap places in the order; and since the
    // order runs from one end of the circle to the other, a swap across the
    // wrap threw the whole schedule, which the ball answered by jumping
    // backwards. It reads as a jitter before it sets off.
    if (!view._plan) {
      for (const a of at) a.angle = Math.atan2(a.p.y - cy, a.p.x - cx);
      at.sort((u, v) => u.angle - v.angle);
      const shots = new Map();
      let fired = 0, total = 0;
      for (const a of at) total += a.dots;
      for (const a of at) {
        for (let k = 0; k < a.dots; k++) shots.set(`${a.id}:${k}`, fired++ / total);
      }
      view._plan = shots;
    }
    const shots = view._plan;

    const s = this.SUPPLY;
    const run = Math.max(0, view.since - s.lead) / s.sweep;
    const first = -Math.PI;             // the leftmost point, and it stays there

    return {
      shot: (id, k) => {
        const when = shots.get(`${id}:${k}`);
        if (when === undefined || run < when) return null;
        const from = first + when * Math.PI * 2;
        const flight = Math.min(1, (run - when) * s.sweep / s.flight);
        return {
          x: cx + Math.cos(from) * ring,
          y: cy + Math.sin(from) * ring,
          in: flight * flight * (3 - 2 * flight)
        };
      },
      draw: (ctx) => {
        const left = 1 - Math.min(1, Math.max(0, run));
        if (left <= 0) return;
        const bx = cx + Math.cos(first + Math.min(1, run) * Math.PI * 2) * ring;
        const by = cy + Math.sin(first + Math.min(1, run) * Math.PI * 2) * ring;
        const r = 3 + 26 * left;
        ctx.globalAlpha = 1;
        ctx.beginPath(); ctx.arc(bx, by, r, 0, Math.PI * 2); ctx.fill();
      }
    };
  },

  /**
   * The staking, as it looks to an agent's own pile.
   *
   * A pile loses what it sends when the tokens set off and gains what it is
   * sent when they land — not one smooth slide from the old balance to the
   * new one, which had piles filling up before anything had reached them and
   * emptying tokens that were still sitting in orbit. The two ends are read
   * off the same schedule the dots on the links are drawn from: they leave
   * over the first quarter of the crossing and arrive over the last.
   *
   * What stays home — an agent's stake on itself — is in neither, so at the
   * end of it the holding is exactly everything staked on the agent.
   */
  _staking(view) {
    const stage = view.stage;
    const out = new Map(), inc = new Map();
    for (const [to, from, amount] of (stage.marks.staked || [])) {
      if (from === to) continue;
      out.set(from, (out.get(from) || 0) + amount);
      inc.set(to, (inc.get(to) || 0) + amount);
    }
    const g = this.GAME;
    const run = (view.since - g.hold) / g.travel;
    const at = Math.max(0, Math.min(1, run));
    const eased = at * at * (3 - 2 * at);
    const ramp = (a, b) => Math.max(0, Math.min(1, (eased - a) / (b - a)));
    const gone = ramp(0, g.spread);
    const come = ramp(1 - g.spread, 1);
    return {
      at: eased,
      raw: run,
      held: (id, had) =>
        had - (out.get(id) || 0) * gone + (inc.get(id) || 0) * come
    };
  },

  _tokens(ctx, view, place, time, w, h) {
    // The opening step is a single agent and what it is made of; the pile it
    // happens to be holding is the next step's subject.
    if (view.effects.has('bare')) return;
    const stage = view.stage;
    // Falls back to the current stage only if nobody has set a run-wide scale.
    const ref = Math.max(2, view.tokenRef || Math.max(1, ...stage.tokens));
    const full = Math.log(ref);

    // How many dots an agent's pile is drawn with. Anchored at both ends: one
    // token is one dot, the reference is fifteen. Never more dots than tokens,
    // which makes the small piles literally countable — the log only starts
    // compressing once there are more than about six, which is also about
    // where counting them stops working.
    const count = held => Math.max(1, Math.min(this.TOKEN_DOTS, held,
      1 + Math.round((this.TOKEN_DOTS - 1) * Math.log(held) / full)));

    // The supply arriving, on the step that introduces it. It comes from
    // nowhere in the world, so it comes from a ball outside the graph that
    // works its way round and fires as it goes, shrinking as it empties.
    const supply = view.effects.has('arrive')
      ? this._supply(view, place, count, w, h) : null;
    // Everyone stakes everything, so during the game a pile drains as it
    // leaves and fills with whatever was aimed at it.
    const moving = view.effects.has('stakes') ? this._staking(view) : null;
    // A child holds nothing until what it was given has crossed the link, and
    // its parent is still holding it until then.
    const given = view.effects.has('inherit') ? this._inheritance(view) : null;

    ctx.save();
    ctx.fillStyle = this.ink.good;
    ctx.shadowColor = 'rgba(127, 212, 160, 0.85)';
    ctx.shadowBlur = 7;

    stage.ids.forEach((id, i) => {
      const p = place(id);
      if (!p) return;
      let held = stage.tokens[i];
      if (moving) held = Math.round(moving.held(id, held));
      if (given) held = Math.round(given.held(id, held));
      if (held <= 0) return;                 // nothing to hold, nothing in orbit
      const dots = count(held);

      // Spread evenly, so the ring says how many there are at a glance rather
      // than needing to be counted through gaps and clumps. Each agent's ring
      // starts at its own angle, or every ring in the graph would line up.
      const orbit = p.r * 1.5;
      const start = this._hash(id) * Math.PI * 2;
      const size = Math.max(1.9, Math.min(3.1, (Math.PI * 2 * orbit / dots) * 0.42));

      for (let k = 0; k < dots; k++) {
        const angle = start + (k / dots) * Math.PI * 2 + time * 0.45;
        const toX = p.x + Math.cos(angle) * orbit;
        const toY = p.y + Math.sin(angle) * orbit;

        let x = toX, y = toY, alpha = p.alpha * 0.9;
        if (supply) {
          const shot = supply.shot(id, k, dots);
          if (!shot) continue;               // not fired yet
          x = shot.x + (toX - shot.x) * shot.in;
          y = shot.y + (toY - shot.y) * shot.in;
          alpha = p.alpha * Math.min(1, shot.in * 4);
        }
        ctx.globalAlpha = alpha;
        ctx.beginPath();
        ctx.arc(x, y, size, 0, Math.PI * 2);
        ctx.fill();
      }
    });

    if (supply) supply.draw(ctx);
    ctx.restore();
    ctx.globalAlpha = 1;
  },

  /**
   * The mutation sweeping across the graph, one agent at a time.
   *
   * Every brain is jittered in the same instant. Drawn that way it is a single
   * flicker across the whole graph and reads as nothing, so it is dealt out
   * one agent at a time instead — and each one stays marked once its turn has
   * passed, so what builds up is the fact that every brain was touched, not
   * just the one currently being touched.
   *
   * Timed from when the step was opened rather than from the clock, so it
   * always starts at the first agent and does not begin halfway through.
   */
  mutating(view) {
    if (!view.stage || !view.effects.has('mutate')) {
      view._mutated = null;
      return;
    }
    const ids = view.stage.ids;
    const done = Math.min(ids.length, Math.floor(view.since / 0.0275));
    if (view._mutated && view._mutated.size === done) return;

    view._mutated = new Map();
    for (let i = 0; i < done; i++) {
      // Which unit of that brain took the hit. Fixed per agent so it does not
      // flicker from frame to frame once it has been chosen.
      view._mutated.set(ids[i], (ids[i] * 3 + i) % 7);
    }
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
    for (const id of view.stage.ids) {
      const near = adj.get(id) || [];
      const targets = [...near, id];                 // itself, last

      // Every eye holds a look for the same length of time, and every eye
      // starts its own somewhere else in that stretch. Switching them all on
      // one clock made the whole graph flick at once, which reads as a cut
      // rather than as forty agents each reading their own neighbourhood.
      const when = seconds / this.GAZE + this._hash(id * 31 + 11);
      const turn = Math.floor(when);
      view._gaze.set(id, {
        at: targets[Math.abs(turn + id) % targets.length],
        through: when - turn                         // 0 to 1 across one look
      });
    }
  }
};
