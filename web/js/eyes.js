/*
 * The front page's other backdrop: a network of eyes, grown out of the
 * screen itself, round one great eye that looks back at you.
 *
 * The mark is six agents, each one an eye, the middle one joined to the rest;
 * this is the mark come alive. The great eye stands where the mark stands,
 * above the title, and stares straight out: its pupil sits in the middle of
 * the ball and will not keep still, trembling a little every frame and now
 * and then darting. Around it the other eyes hang in a loose web, drifting on
 * slow currents; they glance towards the great eye, look away, and blink.
 * None of them looks at you. The threads between them are straight, and
 * tremble a little at random, as a wire does.
 *
 * It grows out of the screen's own pixels. At the very edge the web is a grid
 * of dots a pixel across; going in, the spacing and the eyes grow by the same
 * factor at every step, as a coastline repeats itself at every scale, so the
 * dots draw apart, take on threads, open into eyes, and reach their full size
 * a third of the way in. It is faint at the edge and more solid towards the
 * great eye.
 *
 * Cheap on purpose. The many dots and small eyes near the edge, and the
 * threads between them, never move: they are drawn once into a picture kept
 * aside, and only the few hundred eyes that drift and blink, with their
 * threads, are drawn again, at most thirty times a second, and only from
 * Home.tick — so only while the front page is on screen. For a reader who
 * asks for less motion it is one still picture, drawn again only when the
 * page changes size or colour.
 */
const Eyes = {
  /** Draws a second, at most: a backdrop, like the recorded world. */
  MAX_FPS: 30,

  /** Seconds the web takes to come up out of the glass on arrival. */
  FADE_IN: 1.8,

  /** The finest spacing, at the very edge, in pixels: a dot at every fourth pixel. */
  GAP_MIN: 4,

  /** Below this radius an eye is a dot; from it, a small eye; from WHOLE, a whole eye. */
  DOT: 1.6,

  /** From this radius a dot takes threads, faint at first, as strong as an eye's by WHOLE. */
  THREAD: 1,

  /** From this radius an eye is whole: it drifts, looks about and blinks, and is drawn every frame. */
  WHOLE: 3.5,

  canvas: null,
  host: null,
  anchor: null,
  active: false,
  time: 0,
  nodes: [],
  links: [],
  main: null,
  still: null,

  init(canvas, host, anchor) {
    this.canvas = canvas;
    this.host = host;
    this.anchor = anchor;
  },

  /** Shown: laid out for this screen, from the first frame. */
  start() {
    if (!this.canvas) return;
    this.active = true;
    this.time = 0;
    this._since = 1;              // draw on the first tick
    this.layout();
    this.draw();
  },

  /** Left, or the world chosen instead: nothing more is drawn. */
  stop() {
    this.active = false;
    if (this.ctx && this.width) this.ctx.clearRect(0, 0, this.width, this.height);
  },

  /** The screen changed size: a new web for it. */
  resize() {
    if (!this.active) return;
    this.layout();
    this.draw();
  },

  /** The theme changed: the same web in the new inks. */
  repaint() {
    if (!this.active) return;
    this.paintStill();
    this.draw();
  },

  /** Held still for a reader who asked for less motion (Entrance knows). */
  get calm() {
    return typeof Entrance !== 'undefined' && Entrance.calm;
  },

  /**
   * The web for a screen of this size. The screen is cut into square cells,
   * each cut in four wherever it is wider than the spacing wanted at its
   * middle — a little more or less at random, so the steps between sizes do
   * not show as rings — and one eye is put in each cell: on its grid where
   * the cells are pixels, anywhere in its middle where they are wide. Each
   * eye with threads is joined to its nearest few, and the great eye to six,
   * as the mark's middle eye is joined to its neighbours. Seeded, so the same
   * screen grows the same web.
   */
  layout() {
    const host = this.host, canvas = this.canvas;
    const width = host.clientWidth, height = host.clientHeight;
    if (!(width > 0 && height > 0)) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    this.ctx = canvas.getContext('2d');
    this.dpr = dpr;
    this.width = width;
    this.height = height;

    // Where the great eye stands: where the mark is, or the middle.
    const box = host.getBoundingClientRect();
    const mark = this.anchor ? this.anchor.getBoundingClientRect() : null;
    const cx = mark && mark.width ? mark.left + mark.width / 2 - box.left : width / 2;
    const cy = mark && mark.height ? mark.top + mark.height / 2 - box.top : height * 0.4;
    const big = Math.max(34, Math.min(96, (mark && mark.width ? mark.width : 140) * 0.42));
    this.main = { x0: cx, y0: cy, r: big, whole: true };

    // The spacing at a point: GAP_MIN at the edge, growing by one factor for
    // every pixel in, until it is full a third of the way across.
    const gapMax = Math.max(54, Math.min(96, Math.min(width, height) / 9));
    this.depth = Math.min(width, height) * 0.36;
    const grow = Math.log(gapMax / this.GAP_MIN) / this.depth;
    const gapAt = (x, y) => Math.min(gapMax, this.GAP_MIN * Math.exp(grow * this.inset(x, y)));

    // The words keep the web out: no eye sits on the title, the subtitle, the
    // ways in or the note, where it would be one more thing to read through.
    const pad = gapMax * 0.3;
    const clear = ['#homeTitle', '#homeSubtitle', '.home-doors', '.home-note', '.home-backdrop']
      .map(selector => host.querySelector(selector))
      .filter(el => el && el.offsetParent !== null)
      .map(el => el.getBoundingClientRect())
      .map(r => ({ left: r.left - box.left - pad, right: r.right - box.left + pad,
                   top: r.top - box.top - pad, bottom: r.bottom - box.top + pad }));
    const onWords = (x, y) => clear.some(r => x > r.left && x < r.right && y > r.top && y < r.bottom);

    const rand = seededRandom(4111);
    const nodes = [];
    const side = 2 ** Math.ceil(Math.log2(Math.max(width, height)));
    const cells = [[0, 0, side]];
    while (cells.length) {
      const [x, y, s] = cells.pop();
      if (x >= width || y >= height) continue;
      const want = gapAt(Math.min(x + s / 2, width), Math.min(y + s / 2, height));
      if (s > this.GAP_MIN && s > want * (0.7 + 0.6 * rand())) {
        const h = s / 2;
        cells.push([x, y, h], [x + h, y, h], [x, y + h, h], [x + h, y + h, h]);
        continue;
      }
      const loose = s >= 16 ? 0.6 : 0;
      const px = x + s * (0.5 + loose * (rand() - 0.5)), py = y + s * (0.5 + loose * (rand() - 0.5));
      if (px >= width || py >= height) continue;
      if (Math.hypot(px - cx, py - cy) < big * 2.6 || onWords(px, py)) continue;
      const r = Math.min(gapMax * 0.18, s * (0.13 + 0.06 * rand()));
      const node = { x0: px, y0: py, s, r, whole: r >= this.WHOLE };
      if (node.whole) {
        // Two slow currents on each axis, at speeds and phases of its own,
        // the smaller the eye the smaller its drift.
        const drift = Math.min(1, r / 8);
        Object.assign(node, {
          ax: (2 + 6 * rand()) * drift, ay: (2 + 6 * rand()) * drift,
          fx: 0.05 + 0.13 * rand(), fy: 0.05 + 0.13 * rand(),
          px: rand() * Math.PI * 2, py: rand() * Math.PI * 2,
          // Where it looks: towards the great eye, swinging off it and back.
          swing: 0.4 + 0.9 * rand(), gaze: rand() * Math.PI * 2, glance: 0.12 + 0.25 * rand(),
          // When it blinks: every few seconds, a different few for each.
          blink: 3 + 7 * rand(), blinkAt: rand() * 9
        });
      }
      nodes.push(node);
    }
    this.nodes = nodes;

    // Threads: each eye but the finest dots to its three nearest within
    // twice its spacing, the great eye to its six nearest. Found on a grid of
    // the eyes with threads, so a big screen's thousands stay quick.
    const all = [this.main, ...nodes];
    const threaded = all.map((n, i) => i).filter(i => i === 0 || all[i].r >= this.THREAD);
    const CELL = 16;
    const grid = new Map();
    const cellOf = (x, y) => `${Math.floor(x / CELL)},${Math.floor(y / CELL)}`;
    for (const i of threaded) {
      const k = cellOf(all[i].x0, all[i].y0);
      if (!grid.has(k)) grid.set(k, []);
      grid.get(k).push(i);
    }
    const links = [];
    const seen = new Set();
    for (const i of threaded) {
      const n = all[i];
      const reach = i === 0 ? Math.max(width, height) : n.s * 2.1;
      const span = Math.ceil(reach / CELL);
      const gx = Math.floor(n.x0 / CELL), gy = Math.floor(n.y0 / CELL);
      const found = [];
      for (let a = gx - span; a <= gx + span; a++) {
        for (let b = gy - span; b <= gy + span; b++) {
          for (const j of grid.get(`${a},${b}`) || []) {
            if (j === i) continue;
            const d = Math.hypot(all[j].x0 - n.x0, all[j].y0 - n.y0);
            if (d < reach) found.push([j, d]);
          }
        }
      }
      found.sort((p, q) => p[1] - q[1]);
      for (const [j, d] of found.slice(0, i === 0 ? 6 : 3)) {
        const id = i < j ? `${i}-${j}` : `${j}-${i}`;
        if (seen.has(id)) continue;
        seen.add(id);
        // A thread that moves with an eye is drawn every frame, and trembles
        // by up to a twenty-fifth of its length; one between still eyes is
        // drawn once, straight.
        const moves = i === 0 || all[i].whole || all[j].whole;
        links.push({ a: i, b: j, moves, amp: Math.min(3, d * 0.04), w1: 0, w2: 0,
                     grown: this.grown(Math.min(all[i].r, all[j].r)) });
      }
    }
    this.links = links;

    // How far the web reaches before it has faded into the glass: the
    // farthest corner from the great eye.
    this.reach = Math.max(...[[0, 0], [width, 0], [0, height], [width, height]]
      .map(([x, y]) => Math.hypot(x - cx, y - cy)));
    this.dart = { x: 0, y: 0, next: 0 };
    this.paintStill();
  },

  /**
   * How strong a thread is for the size of the smaller eye it joins: next to
   * nothing between the finest dots, full between whole eyes, so the web
   * condenses out of the dots rather than starting at a line.
   */
  grown(r) {
    const k = Math.max(0, Math.min(1, (r - this.THREAD) / (this.WHOLE - this.THREAD)));
    return 0.08 + 0.92 * k * k * (3 - 2 * k);
  },

  /** How far a point lies inside the screen: its distance to the nearest edge. */
  inset(x, y) {
    return Math.max(0, Math.min(x, this.width - x, y, this.height - y));
  },

  /** One animation frame from Home, while the front page shows the eyes. */
  tick(dt) {
    if (!this.active || this.calm) return;
    this.time += dt;
    this._since = (this._since || 0) + dt;
    if (this._since < 1 / this.MAX_FPS) return;
    this._since = 0;
    this.draw();
  },

  /** Where an eye is now: its place, carried by its two currents if it is whole. */
  place(n, t) {
    if (n === this.main) return { x: n.x0 + Math.sin(t * 0.35) * 1.5, y: n.y0 + Math.cos(t * 0.29) * 1.5 };
    if (!n.whole) return { x: n.x0, y: n.y0 };
    const T = Math.PI * 2 * t;
    return {
      x: n.x0 + n.ax * Math.sin(T * n.fx + n.px) + n.ax * 0.45 * Math.sin(T * n.fx * 2.3 + n.py),
      y: n.y0 + n.ay * Math.sin(T * n.fy + n.py) + n.ay * 0.45 * Math.sin(T * n.fy * 1.7 + n.px)
    };
  },

  /**
   * How solid something is at a point: faint at the edges of the screen,
   * where the web is the screen's own pixels, and more solid inwards and the
   * nearer the great eye.
   */
  solid(x, y) {
    const smooth = k => { const c = Math.max(0, Math.min(1, k)); return c * c * (3 - 2 * c); };
    const edge = 0.3 + 0.7 * smooth(this.inset(x, y) / this.depth);
    const near = smooth(1 - Math.hypot(x - this.main.x0, y - this.main.y0) / this.reach);
    return edge * (0.4 + 0.6 * near);
  },

  /** The inks of the eyes, as the theme has them now. */
  inks() {
    return { ball: Ink.of('eyeBall'), iris: Ink.of('eyeIris'), pupil: Ink.of('eyePupil'),
             rim: Ink.of('eyeRim'), link: Ink.of('eyeLink') };
  },

  /**
   * The part of the web that never moves — the dots, the small eyes and the
   * threads between them — drawn once, at full strength, into a picture kept
   * aside; draw() lays it down every frame, faded in as the web comes up.
   */
  paintStill() {
    if (!this.ctx) return;
    const still = this.still || (this.still = document.createElement('canvas'));
    still.width = this.canvas.width;
    still.height = this.canvas.height;
    const g = still.getContext('2d');
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.width, this.height);
    const ink = this.inks();
    const all = [this.main, ...this.nodes];
    const LEVELS = 8;
    const level = a => Math.min(LEVELS - 1, Math.floor(a * LEVELS));
    // Things of about one strength in one path, filled or stroked at once.
    const shade = (items, paint, ink) => {
      const buckets = Array.from({ length: LEVELS }, () => []);
      for (const item of items) buckets[level(item.alpha)].push(item);
      buckets.forEach((list, k) => {
        if (!list.length) return;
        g.globalAlpha = (k + 0.5) / LEVELS;
        g.beginPath();
        for (const item of list) paint(item);
        ink();
      });
    };

    // The still threads, straight.
    g.strokeStyle = ink.link;
    g.lineWidth = 0.8;
    shade(this.links.filter(l => !l.moves).map(l => {
      const p = all[l.a], q = all[l.b];
      return { p, q, alpha: Math.min(this.solid(p.x0, p.y0), this.solid(q.x0, q.y0)) * 0.5 * l.grown };
    }), ({ p, q }) => { g.moveTo(p.x0, p.y0); g.lineTo(q.x0, q.y0); }, () => g.stroke());

    // The dots: the screen's own pixels, in the iris's colour.
    g.fillStyle = ink.iris;
    shade(this.nodes.filter(n => n.r < this.DOT).map(n => ({ n, alpha: Math.min(1, this.solid(n.x0, n.y0) * 1.6) })),
          ({ n }) => { const d = Math.max(1, Math.round(n.r * 2)); g.rect(Math.round(n.x0 - d / 2), Math.round(n.y0 - d / 2), d, d); },
          () => g.fill());

    // The small eyes: a ball and a pupil, nothing more at this size.
    const small = this.nodes.filter(n => n.r >= this.DOT && !n.whole).map(n => ({ n, alpha: this.solid(n.x0, n.y0) }));
    g.fillStyle = ink.ball;
    shade(small, ({ n }) => { g.moveTo(n.x0 + n.r, n.y0); g.arc(n.x0, n.y0, n.r, 0, Math.PI * 2); }, () => g.fill());
    g.fillStyle = ink.pupil;
    shade(small, ({ n }) => { g.moveTo(n.x0 + n.r * 0.45, n.y0); g.arc(n.x0, n.y0, n.r * 0.45, 0, Math.PI * 2); },
          () => g.fill());
    g.globalAlpha = 1;
  },

  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.main) return;
    const calm = this.calm;
    // A still picture is the web a few seconds in, fully grown.
    const t = calm ? 6 : this.time;
    const rise = calm ? 1 : Math.min(1, t / this.FADE_IN);
    const fade = rise * rise * (3 - 2 * rise);
    const ink = this.inks();

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    if (this.still) {
      ctx.globalAlpha = fade;
      ctx.drawImage(this.still, 0, 0, this.width, this.height);
    }
    const all = [this.main, ...this.nodes];
    const at = all.map(n => this.place(n, t));
    const alpha = i => this.solid(at[i].x, at[i].y) * fade;

    // ---- the threads that move, in a few strengths so each is one stroke call ----
    ctx.strokeStyle = ink.link;
    ctx.lineWidth = 1;
    const LEVELS = 8;
    const buckets = Array.from({ length: LEVELS }, () => []);
    for (const link of this.links) {
      if (!link.moves) continue;
      // The tremble: two points along the thread, each pushed sideways by a
      // little more or less every frame and drawn back towards the line.
      if (!calm) {
        link.w1 = Math.max(-link.amp, Math.min(link.amp, link.w1 * 0.9 + (Math.random() - 0.5) * link.amp * 0.5));
        link.w2 = Math.max(-link.amp, Math.min(link.amp, link.w2 * 0.9 + (Math.random() - 0.5) * link.amp * 0.5));
      }
      const a = Math.min(alpha(link.a), alpha(link.b)) * 0.5 * link.grown;
      if (a < 0.015) continue;
      buckets[Math.min(LEVELS - 1, Math.floor(a / 0.5 * LEVELS))].push(link);
    }
    buckets.forEach((list, level) => {
      if (!list.length) return;
      ctx.globalAlpha = ((level + 0.5) / LEVELS) * 0.5;
      ctx.beginPath();
      for (const link of list) {
        const p = at[link.a], q = at[link.b];
        const dx = q.x - p.x, dy = q.y - p.y;
        const length = Math.hypot(dx, dy) || 1;
        const nx = -dy / length, ny = dx / length;
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + dx / 3 + nx * link.w1, p.y + dy / 3 + ny * link.w1);
        ctx.lineTo(p.x + dx * 2 / 3 + nx * link.w2, p.y + dy * 2 / 3 + ny * link.w2);
        ctx.lineTo(q.x, q.y);
      }
      ctx.stroke();
    });

    // ---- the whole eyes ----
    for (let i = 1; i < all.length; i++) {
      const n = all[i];
      if (!n.whole) continue;
      const a = alpha(i);
      if (a < 0.02) continue;
      const p = at[i];
      // Towards the great eye, swinging off it now and then.
      const toward = Math.atan2(at[0].y - p.y, at[0].x - p.x);
      const look = toward + n.swing * Math.sin(t * n.glance * Math.PI * 2 + n.gaze);
      // A blink is a quick close and open, a few times a minute.
      const phase = (t + n.blinkAt) % n.blink;
      const open = calm ? 1 : (phase < 0.16 ? Math.abs(Math.cos(Math.PI * phase / 0.16)) : 1);
      this.eye(ctx, p.x, p.y, n.r, look, 0.3, open, a, ink);
    }

    // ---- the great eye, looking at you ----
    const m = this.main, p = at[0];
    // A tremor every frame, and every second or so a dart to a new resting
    // place: an eye that is trying to hold still and cannot.
    let jx = 0, jy = 0;
    if (!calm) {
      if (t >= this.dart.next) {
        const a = Math.random() * Math.PI * 2;
        const d = Math.random() * m.r * 0.05;
        this.dart = { x: Math.cos(a) * d, y: Math.sin(a) * d, next: t + 0.6 + Math.random() * 1.3 };
      }
      jx = this.dart.x + (Math.random() - 0.5) * m.r * 0.04;
      jy = this.dart.y + (Math.random() - 0.5) * m.r * 0.04;
    }
    const breathe = calm ? 1 : 1 + 0.05 * Math.sin(t * 1.7);

    // A faint light behind it, so it sits in the web rather than on it.
    ctx.globalAlpha = 0.22 * fade;
    const glow = ctx.createRadialGradient(p.x, p.y, m.r * 0.8, p.x, p.y, m.r * 2.4);
    glow.addColorStop(0, ink.iris);
    glow.addColorStop(1, Ink.alpha(ink.iris, 0));
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(p.x, p.y, m.r * 2.4, 0, Math.PI * 2); ctx.fill();

    this.eye(ctx, p.x, p.y, m.r, 0, 0, 1, fade, ink, { jx, jy, breathe, spokes: true });
    ctx.globalAlpha = 1;
  },

  /**
   * One eye: the ball, its rim, the iris and the pupil, looking along `look`
   * by `off` of the ball's radius; `open` closes it towards a line. The
   * great eye adds the iris's spokes, a tremor (`jx`, `jy`) and a pupil that
   * swells and shrinks (`breathe`).
   */
  eye(ctx, x, y, r, look, off, open, alpha, ink, { jx = 0, jy = 0, breathe = 1, spokes = false } = {}) {
    const squash = Math.max(0.06, open);
    ctx.globalAlpha = alpha;
    ctx.fillStyle = ink.ball;
    ctx.beginPath(); ctx.ellipse(x, y, r, r * squash, 0, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = ink.rim;
    ctx.lineWidth = Math.max(0.8, r * 0.06);
    ctx.globalAlpha = alpha * 0.55;
    ctx.stroke();
    if (open < 0.25) return;
    ctx.globalAlpha = alpha;

    const ix = x + Math.cos(look) * r * off + jx;
    const iy = y + Math.sin(look) * r * off * squash + jy;
    const iris = r * 0.6, pupil = r * 0.31 * breathe;
    ctx.fillStyle = ink.iris;
    ctx.beginPath(); ctx.ellipse(ix, iy, iris, iris * squash, 0, 0, Math.PI * 2); ctx.fill();

    if (spokes) {
      ctx.globalAlpha = alpha * 0.32;
      ctx.strokeStyle = ink.pupil;
      ctx.lineWidth = Math.max(0.6, r * 0.03);
      ctx.beginPath();
      for (let k = 0; k < 14; k++) {
        const a = (k / 14) * Math.PI * 2 + 0.2;
        ctx.moveTo(ix + Math.cos(a) * pupil * 1.15, iy + Math.sin(a) * pupil * 1.15);
        ctx.lineTo(ix + Math.cos(a) * iris * 0.93, iy + Math.sin(a) * iris * 0.93);
      }
      ctx.stroke();
      ctx.globalAlpha = alpha;
    }

    ctx.fillStyle = ink.pupil;
    ctx.beginPath(); ctx.ellipse(ix, iy, pupil, pupil * squash, 0, 0, Math.PI * 2); ctx.fill();

    // The light on its surface stays where the light is, whatever the pupil does.
    ctx.fillStyle = ink.ball;
    ctx.globalAlpha = alpha * 0.85;
    ctx.beginPath(); ctx.arc(x - r * 0.26, y - r * 0.28 * squash, r * 0.09, 0, Math.PI * 2); ctx.fill();
  }
};
