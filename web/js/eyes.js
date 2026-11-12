/*
 * The front page's other backdrop: a network of eyes, grown out of the
 * screen itself, round one great eye that looks back at you.
 *
 * The mark is six agents, each one an eye, the middle one joined to the rest;
 * this is the mark come alive. The great eye stands where the mark stands,
 * above the title, and stares straight out: its pupil sits in the middle of
 * the ball and will not keep still, trembling a little every frame and now
 * and then darting. Around it the other eyes hang in a loose web, drifting on
 * slow currents, the threads between them flexing as they go; they glance
 * towards the great eye, look away, and blink. None of them looks at you.
 *
 * It is drawn as though it were part of the glass: nothing at the edges of
 * the screen, growing more solid towards the middle, so the web seems to come
 * up out of the border rather than lie on top of it.
 *
 * Cheap on purpose. About a hundred eyes and two hundred threads, positions
 * worked out from a few sines rather than simulated, drawn at most thirty
 * times a second, and only from Home.tick — so only while the front page is
 * on screen. For a reader who asks for less motion it is one still picture,
 * drawn again only when the page changes size or colour.
 */
const Eyes = {
  /** Draws a second, at most: a backdrop, like the recorded world. */
  MAX_FPS: 30,

  /** Seconds the web takes to come up out of the glass on arrival. */
  FADE_IN: 1.8,

  canvas: null,
  host: null,
  anchor: null,
  active: false,
  time: 0,
  nodes: [],
  links: [],
  main: null,

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
    if (this.active) this.draw();
  },

  /** Held still for a reader who asked for less motion (Entrance knows). */
  get calm() {
    return typeof Entrance !== 'undefined' && Entrance.calm;
  },

  /**
   * The web for a screen of this size. Eyes are scattered by throwing darts
   * and keeping those far enough from every eye already kept, which spaces
   * them evenly without lining them up; each is joined to its nearest few,
   * and the great eye to six, as the mark's middle eye is joined to its
   * neighbours. Seeded, so the same screen grows the same web.
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
    this.main = { x0: cx, y0: cy, r: big };

    const rand = seededRandom(4111);
    const gap = Math.max(54, Math.min(96, Math.min(width, height) / 9));
    const want = Math.min(150, Math.floor(width * height / (gap * gap * 1.25)));
    const nodes = [];
    const cell = gap;
    const grid = new Map();
    const key = (x, y) => `${Math.floor(x / cell)},${Math.floor(y / cell)}`;
    const near = (x, y, d) => {
      const gx = Math.floor(x / cell), gy = Math.floor(y / cell);
      for (let i = gx - 1; i <= gx + 1; i++) {
        for (let j = gy - 1; j <= gy + 1; j++) {
          for (const n of grid.get(`${i},${j}`) || []) {
            if (Math.hypot(n.x0 - x, n.y0 - y) < d) return true;
          }
        }
      }
      return false;
    };
    // The words keep the web out: no eye sits on the title, the subtitle, the
    // ways in or the note, where it would be one more thing to read through.
    const pad = gap * 0.3;
    const clear = ['#homeTitle', '#homeSubtitle', '.home-doors', '.home-note', '.home-backdrop']
      .map(selector => host.querySelector(selector))
      .filter(el => el && el.offsetParent !== null)
      .map(el => el.getBoundingClientRect())
      .map(r => ({ left: r.left - box.left - pad, right: r.right - box.left + pad,
                   top: r.top - box.top - pad, bottom: r.bottom - box.top + pad }));
    const onWords = (x, y) => clear.some(r => x > r.left && x < r.right && y > r.top && y < r.bottom);
    for (let tries = 0; tries < want * 30 && nodes.length < want; tries++) {
      const x = rand() * width, y = rand() * height;
      if (Math.hypot(x - cx, y - cy) < big * 2.6) continue;
      if (onWords(x, y) || near(x, y, gap * 0.82)) continue;
      const node = {
        x0: x, y0: y,
        r: gap * (0.1 + 0.08 * rand()),
        // Two slow currents on each axis, at speeds and phases of its own.
        ax: 2 + 6 * rand(), ay: 2 + 6 * rand(),
        fx: 0.05 + 0.13 * rand(), fy: 0.05 + 0.13 * rand(),
        px: rand() * Math.PI * 2, py: rand() * Math.PI * 2,
        // Where it looks: towards the great eye, swinging off it and back.
        swing: 0.4 + 0.9 * rand(), gaze: rand() * Math.PI * 2, glance: 0.12 + 0.25 * rand(),
        // When it blinks: every few seconds, a different few for each.
        blink: 3 + 7 * rand(), blinkAt: rand() * 9
      };
      nodes.push(node);
      const k = key(x, y);
      if (!grid.has(k)) grid.set(k, []);
      grid.get(k).push(node);
    }
    this.nodes = nodes;

    // Each eye to its three nearest, and the great eye to its six.
    const links = [];
    const seen = new Set();
    const join = (a, b) => {
      const id = a < b ? `${a}-${b}` : `${b}-${a}`;
      if (seen.has(id)) return;
      seen.add(id);
      links.push({ a, b, bend: (rand() - 0.5) * 0.5, f: 0.05 + 0.1 * rand(), p: rand() * Math.PI * 2 });
    };
    const all = [this.main, ...nodes];
    all.forEach((n, i) => {
      const nearest = all.map((m, j) => [j, Math.hypot(m.x0 - n.x0, m.y0 - n.y0)])
        .filter(([j]) => j !== i)
        .sort((p, q) => p[1] - q[1])
        .slice(0, i === 0 ? 6 : 3)
        .filter(([, d]) => i === 0 || d < gap * 2.1);
      for (const [j] of nearest) join(i, j);
    });
    this.links = links;

    // How far the web reaches before it has faded into the glass: the
    // farthest corner from the great eye.
    this.reach = Math.max(...[[0, 0], [width, 0], [0, height], [width, height]]
      .map(([x, y]) => Math.hypot(x - cx, y - cy)));
    this.dart = { x: 0, y: 0, next: 0 };
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

  /** Where an eye is now: its place, carried by its two currents. */
  place(n, t) {
    if (n === this.main) return { x: n.x0 + Math.sin(t * 0.35) * 1.5, y: n.y0 + Math.cos(t * 0.29) * 1.5 };
    const T = Math.PI * 2 * t;
    return {
      x: n.x0 + n.ax * Math.sin(T * n.fx + n.px) + n.ax * 0.45 * Math.sin(T * n.fx * 2.3 + n.py),
      y: n.y0 + n.ay * Math.sin(T * n.fy + n.py) + n.ay * 0.45 * Math.sin(T * n.fy * 1.7 + n.px)
    };
  },

  /**
   * How solid something is at a point: nothing at all at any edge of the
   * screen, as though the web were in the glass and the frame hid its ends,
   * and more of it the nearer the great eye.
   */
  solid(x, y) {
    const smooth = k => { const c = Math.max(0, Math.min(1, k)); return c * c * (3 - 2 * c); };
    const edge = smooth(Math.min(x, this.width - x, y, this.height - y)
                        / (Math.min(this.width, this.height) * 0.32));
    const near = smooth(1 - Math.hypot(x - this.main.x0, y - this.main.y0) / this.reach);
    return edge * (0.4 + 0.6 * near);
  },

  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.main) return;
    const calm = this.calm;
    // A still picture is the web a few seconds in, fully grown.
    const t = calm ? 6 : this.time;
    const rise = calm ? 1 : Math.min(1, t / this.FADE_IN);
    const fade = rise * rise * (3 - 2 * rise);
    const ink = {
      ball: Ink.of('eyeBall'), iris: Ink.of('eyeIris'), pupil: Ink.of('eyePupil'),
      rim: Ink.of('eyeRim'), link: Ink.of('eyeLink')
    };

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    const all = [this.main, ...this.nodes];
    const at = all.map(n => this.place(n, t));
    const alpha = at.map(p => this.solid(p.x, p.y) * fade);

    // ---- the threads, in a few strengths so each is one stroke call ----
    ctx.strokeStyle = ink.link;
    ctx.lineWidth = 1;
    const LEVELS = 8;
    const buckets = Array.from({ length: LEVELS }, () => []);
    for (const link of this.links) {
      const a = Math.min(alpha[link.a], alpha[link.b]) * 0.5;
      if (a < 0.015) continue;
      buckets[Math.min(LEVELS - 1, Math.floor(a / 0.5 * LEVELS))].push(link);
    }
    buckets.forEach((list, level) => {
      if (!list.length) return;
      ctx.globalAlpha = ((level + 0.5) / LEVELS) * 0.5;
      ctx.beginPath();
      for (const link of list) {
        const p = at[link.a], q = at[link.b];
        const mx = (p.x + q.x) / 2, my = (p.y + q.y) / 2;
        const dx = q.x - p.x, dy = q.y - p.y;
        const flex = link.bend + 0.12 * Math.sin(t * Math.PI * 2 * link.f + link.p);
        ctx.moveTo(p.x, p.y);
        ctx.quadraticCurveTo(mx - dy * flex, my + dx * flex, q.x, q.y);
      }
      ctx.stroke();
    });

    // ---- the small eyes ----
    for (let i = 1; i < all.length; i++) {
      if (alpha[i] < 0.02) continue;
      const n = all[i], p = at[i];
      // Towards the great eye, swinging off it now and then.
      const toward = Math.atan2(at[0].y - p.y, at[0].x - p.x);
      const look = toward + n.swing * Math.sin(t * n.glance * Math.PI * 2 + n.gaze);
      // A blink is a quick close and open, a few times a minute.
      const phase = (t + n.blinkAt) % n.blink;
      const open = calm ? 1 : (phase < 0.16 ? Math.abs(Math.cos(Math.PI * phase / 0.16)) : 1);
      this.eye(ctx, p.x, p.y, n.r, look, 0.3, open, alpha[i], ink);
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
