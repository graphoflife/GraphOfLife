/*
 * The front page's other backdrop: a network of eyes, grown out of the
 * screen itself, round one great eye that looks back at you.
 *
 * The mark is six agents, each one an eye, the middle one joined to the rest;
 * this is the mark come alive. The great eye stands where the mark stands,
 * above the title, and stares straight out: its pupil sits in the middle of
 * the ball and will not keep still, trembling a little every frame and now
 * and then darting. The eye itself holds its place, inside a small clear ring
 * that no other eye enters. None of the others looks at you; they look at the
 * great eye — the small ones calmly, the big ones nervously, in quick glances
 * at it and away. The threads between them are straight, and tremble a little
 * at random, as a wire does.
 *
 * It grows out of the screen's own pixels. At the very edge the web is a grid
 * of dots a pixel across; from there to the clear ring the eyes and the
 * spacing between them grow in a straight line, so the dots draw apart, take
 * on threads, open into eyes, and end as the biggest, next to the great eye.
 * An eye's size is a matter of where it is, not of which eye it is: one that
 * drifts inwards grows, one that drifts out shrinks. The words lie over the
 * web and do not part it; the title carries its own shadow (style.css).
 *
 * It is alive in the middle and still at its rims. The dots at the edge never
 * move, nor does the great eye; every other eye wanders round its place,
 * fastest halfway in and slower towards the edge and the great eye, never so
 * far as to crowd a neighbour or leave a gap. As they wander the threads
 * follow: each eye holds on to its nearest three, so a thread to a neighbour
 * that has drifted off fades away while one to a nearer neighbour fades in.
 *
 * Cheap on purpose. The dots at the edge, and the threads between them, are
 * drawn once into a picture kept aside. The eyes that move — some fifteen
 * hundred on a large screen — are kept in typed arrays, each with a short
 * list of the neighbours it could reach, and drawn with their threads in a
 * few batches, at most thirty times a second, and only from Home.tick — so
 * only while the front page is on screen. For a reader who asks for less
 * motion it is one still picture, drawn again only when the page changes
 * size or colour.
 */
const Eyes = {
  /** Draws a second, at most: a backdrop, like the recorded world. */
  MAX_FPS: 30,

  /** Seconds the web takes to come up out of the glass on arrival. */
  FADE_IN: 1.8,

  /** The finest spacing, at the very edge, in pixels: a dot at every fourth pixel. */
  GAP_MIN: 4,

  /** An eye's radius at the very edge: a dot one pixel across. */
  EDGE: 0.5,

  /** Below this radius an eye is a dot; from it, a small eye; from WHOLE, a whole eye. */
  DOT: 1.6,

  /** From this radius a dot takes threads, faint at first, as strong as an eye's by WHOLE. */
  THREAD: 1,

  /** From this radius an eye is whole: it looks about and blinks. */
  WHOLE: 3.5,

  /** From this radius an eye is drawn on its own, its rim as heavy as its size. */
  LARGE: 13,

  /** The clear ring round the great eye, in radii of the great eye. */
  CLEAR: 1.95,

  /** The eyes just outside the clear ring, the biggest after the great eye, in its radii. */
  NEXT: 0.45,

  /** The spacing there, in radii of the eyes there. */
  ROOM: 5.2,

  /** How far an eye wanders from its place, as a share of the spacing there. */
  WANDER: 0.34,

  /** Seconds a new thread takes to come up, and an old one to fade away. */
  KNIT: 0.8,
  FRAY: 1.2,

  canvas: null,
  host: null,
  anchor: null,
  active: false,
  time: 0,
  n: 0,
  cache: null,

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
   * not show as rings — and each cell offers one place: on its grid where the
   * cells are pixels, anywhere in its middle where they are wide. The places
   * are then taken biggest first, and one too near a place already taken is
   * dropped, so no two eyes crowd each other whatever the cells did. Each eye
   * that takes threads learns its nearest eight once, here, so that choosing
   * its nearest three as it moves is a look down a short list. Seeded, so the
   * same screen grows the same web.
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
    this.cx = cx;
    this.cy = cy;
    this.big = big;

    // The clear ring, and just outside it the biggest of the other eyes, the
    // middles of the nearest a little further out so that their balls only
    // touch it. From there the eyes and the spacing shrink in a straight line
    // to a dot at every fourth pixel on the screen's edge.
    this.clear = big * this.CLEAR;
    this.rNext = big * this.NEXT;
    this.ring = this.clear + this.rNext;
    this.gNext = this.rNext * this.ROOM;

    const rand = seededRandom(4111);
    const TAU = Math.PI * 2;

    // ---- the places on offer, one a cell ----
    const offer = [];
    const side = 2 ** Math.ceil(Math.log2(Math.max(width, height)));
    const cells = [[0, 0, side]];
    while (cells.length) {
      const [x, y, s] = cells.pop();
      if (x >= width || y >= height) continue;
      const want = this.gapAt(this.deep(Math.min(x + s / 2, width), Math.min(y + s / 2, height)));
      if (s > this.GAP_MIN && s > want * (0.7 + 0.6 * rand())) {
        const h = s / 2;
        cells.push([x, y, h], [x + h, y, h], [x, y + h, h], [x + h, y + h, h]);
        continue;
      }
      // A cell the clear ring cuts tries a few times, a little wider each
      // time, for the part of it outside the ring.
      const loose = s >= 16 ? 0.6 : 0;
      for (let attempt = 0; attempt < (loose ? 6 : 1); attempt++) {
        const spread = loose ? loose + 0.06 * attempt : 0;
        const px = x + s * (0.5 + spread * (rand() - 0.5)), py = y + s * (0.5 + spread * (rand() - 0.5));
        if (px >= width || py >= height) break;
        if (Math.hypot(px - cx, py - cy) < this.ring) continue;
        offer.push(px, py);
        break;
      }
    }

    // ---- taken biggest first, none too near another ----
    const offered = offer.length / 2;
    const wants = new Float32Array(offered);
    for (let i = 0; i < offered; i++) wants[i] = this.gapAt(this.deep(offer[2 * i], offer[2 * i + 1]));
    const order = Array.from({ length: offered }, (_, i) => i).sort((a, b) => wants[b] - wants[a]);
    const CELL = 16, gw = Math.ceil(width / CELL) + 1, gh = Math.ceil(height / CELL) + 1;
    let head = new Int32Array(gw * gh).fill(-1);
    let chain = new Int32Array(offered);
    const taken = [];
    for (const i of order) {
      const px = offer[2 * i], py = offer[2 * i + 1], want = wants[i];
      const span = Math.ceil((want * 1.6 + 2) / CELL);
      const gx = Math.floor(px / CELL), gy = Math.floor(py / CELL);
      let free = true;
      for (let b = Math.max(0, gy - span); free && b <= Math.min(gh - 1, gy + span); b++) {
        for (let a = Math.max(0, gx - span); free && a <= Math.min(gw - 1, gx + span); a++) {
          for (let j = head[b * gw + a]; j >= 0; j = chain[j]) {
            if (Math.hypot(offer[2 * j] - px, offer[2 * j + 1] - py) < 0.4 * (want + wants[j])) { free = false; break; }
          }
        }
      }
      if (!free) continue;
      chain[i] = head[gy * gw + gx];
      head[gy * gw + gx] = i;
      taken.push(i);
    }

    // ---- the eyes: the great eye first, then the rest ----
    const n = taken.length + 1;
    const F = () => new Float32Array(n);
    Object.assign(this, {
      n,
      hx: F(), hy: F(),                 // its place
      x: F(), y: F(),                   // where it is now
      r: F(), k: F(), alpha: F(),       // its size, how deep it is, how solid
      grain: F(),                       // its own size beside its neighbours'
      gap: F(),                         // the spacing at its place
      reach: F(),                       // how far it wanders
      still: new Uint8Array(n),         // a dot at the edge: it never moves
      fx: F(), fy: F(), qx: F(), qy: F(),
      phx: new Float64Array(n), phy: new Float64Array(n),
      swell: new Float32Array(8 * n),   // where it sits on each current: a sine and a cosine for each
      gaze: F(), gazeTo: F(), gazeAt: new Float64Array(n),
      blink: F(), blinkAt: F(),
      ix: F(), iy: F(), open: F()       // where the pupil is and how open the eye, this frame
    });

    // The currents: slow waves a screen or so long, two on each axis, so that
    // neighbours sway together and the web moves as one body.
    const length = Math.min(width, height);
    const waves = [0, 1, 2, 3].map(() => {
      const a = rand() * TAU, kk = TAU / (length * (0.5 + 0.5 * rand()));
      return { kx: Math.cos(a) * kk, ky: Math.sin(a) * kk, at: rand() * TAU,
               beat: TAU * (0.03 + 0.04 * rand()) * (rand() < 0.5 ? -1 : 1) };
    });
    this.waves = waves;

    this.hx[0] = this.x[0] = cx;
    this.hy[0] = this.y[0] = cy;
    this.r[0] = big;
    this.k[0] = 1;
    this.alpha[0] = 1;
    this.still[0] = 1;
    taken.forEach((i, at) => {
      const e = at + 1;
      const px = offer[2 * i], py = offer[2 * i + 1];
      const k = this.deep(px, py);
      this.hx[e] = this.x[e] = px;
      this.hy[e] = this.y[e] = py;
      this.k[e] = k;
      this.grain[e] = 0.8 + 0.4 * rand();
      this.gap[e] = this.gapAt(k);
      this.r[e] = this.sizeAt(k, e, px, py);
      this.alpha[e] = this.solid(k);
      this.still[e] = this.r[e] < this.DOT ? 1 : 0;
      // How far it may wander: a share of the spacing, and never so far as
      // to reach the screen's edge or the clear ring.
      const toRing = Math.hypot(px - cx, py - cy) - this.ring;
      this.reach[e] = this.still[e] ? 0 : Math.min(this.WANDER * this.gap[e], 0.6 * this.inset(px, py), 0.6 * toRing);
      // Its own wander: two slow swings on each axis, at speeds of its own.
      this.fx[e] = 0.05 + 0.08 * rand();
      this.fy[e] = 0.05 + 0.08 * rand();
      this.phx[e] = rand() * TAU;
      this.phy[e] = rand() * TAU;
      this.qx[e] = rand() * TAU;
      this.qy[e] = rand() * TAU;
      waves.forEach((wave, j) => {
        const at = wave.kx * px + wave.ky * py + wave.at;
        this.swell[8 * e + 2 * j] = Math.sin(at);
        this.swell[8 * e + 2 * j + 1] = Math.cos(at);
      });
      // When it blinks: every few seconds, a different few for each, and
      // more often the bigger and more nervous it is.
      this.blink[e] = (3 + 7 * rand()) * (1 - 0.5 * this.nerves(this.r[e]));
      this.blinkAt[e] = rand() * 9;
      this.gazeAt[e] = rand() * 2;
    });

    // ---- the threads each eye could take ----
    // Found on a grid of the eyes with threads, so a big screen's thousands
    // stay quick. A dot never moves, so its own three are chosen here once;
    // threads between two dots go into the picture kept aside.
    const threaded = [];
    for (let e = 0; e < n; e++) if (!this.still[e] || e === 0 || this.r[e] >= this.THREAD) threaded.push(e);
    head = new Int32Array(gw * gh).fill(-1);
    chain = new Int32Array(n);
    for (const e of threaded) {
      const c = Math.floor(this.hy[e] / CELL) * gw + Math.floor(this.hx[e] / CELL);
      chain[e] = head[c];
      head[c] = e;
    }
    const pairs = [], index = new Map(), pinned = new Set(), fixed = [];
    const pairOf = (a, b) => {
      const lo = Math.min(a, b), hi = Math.max(a, b), key = lo * n + hi;
      let p = index.get(key);
      if (p === undefined) { p = pairs.length / 2; index.set(key, p); pairs.push(lo, hi); }
      return p;
    };
    const fixedSeen = new Set();
    for (const e of threaded) {
      const far = e === 0 ? this.ring + 2.5 * this.gNext : this.gap[e] * 2.6;
      const span = Math.ceil(far / CELL);
      const gx = Math.floor(this.hx[e] / CELL), gy = Math.floor(this.hy[e] / CELL);
      const found = [];
      for (let b = Math.max(0, gy - span); b <= Math.min(gh - 1, gy + span); b++) {
        for (let a = Math.max(0, gx - span); a <= Math.min(gw - 1, gx + span); a++) {
          for (let j = head[b * gw + a]; j >= 0; j = chain[j]) {
            if (j === e) continue;
            const d = Math.hypot(this.hx[j] - this.hx[e], this.hy[j] - this.hy[e]);
            if (d < far) found.push([j, d]);
          }
        }
      }
      found.sort((p, q) => p[1] - q[1]);
      for (const [j] of found.slice(0, e === 0 ? 14 : 8)) {
        if (!this.still[e] || !this.still[j] || e === 0 || j === 0) pairOf(e, j);
      }
      if (this.still[e] && e !== 0) {
        for (const [j, d] of found.slice(0, 3)) {
          if (d > this.gap[e] * 2.1) break;
          if (this.still[j] && j !== 0) {
            const key = Math.min(e, j) * n + Math.max(e, j);
            if (!fixedSeen.has(key)) { fixedSeen.add(key); fixed.push(e, j); }
          } else {
            pinned.add(pairOf(e, j));
          }
        }
      }
    }
    const P = pairs.length / 2;
    this.pa = new Int32Array(P);
    this.pb = new Int32Array(P);
    for (let p = 0; p < P; p++) { this.pa[p] = pairs[2 * p]; this.pb[p] = pairs[2 * p + 1]; }
    this.pd = new Float32Array(P);
    this.sig = new Float32Array(P);
    this.w1 = new Float32Array(P);
    this.w2 = new Float32Array(P);
    this.want = new Uint8Array(P);
    this.pin = new Uint8Array(P);
    this.from = new Uint8Array(P);    // which end a thread grows from: 0 its first eye, 1 its second
    for (const p of pinned) this.pin[p] = 1;
    this.fixed = Int32Array.from(fixed);
    // Each eye's pairs, in one array: an eye's run starts at adjStart[e].
    const degree = new Int32Array(n + 1);
    for (let p = 0; p < P; p++) { degree[this.pa[p] + 1]++; degree[this.pb[p] + 1]++; }
    for (let e = 0; e < n; e++) degree[e + 1] += degree[e];
    this.adjStart = degree;
    this.adj = new Int32Array(degree[n]);
    const fill = degree.slice(0, n);
    for (let p = 0; p < P; p++) { this.adj[fill[this.pa[p]]++] = p; this.adj[fill[this.pb[p]]++] = p; }

    // Kept between frames, so drawing makes no garbage.
    this.bins = Array.from({ length: 4 }, () => Array.from({ length: 8 }, () => []));
    this.large = [];
    this.dart = { x: 0, y: 0, next: 0 };

    // The first frame shows each eye already holding its threads.
    this.move(0, true);
    this.paintStill();
  },

  /**
   * How deep a point lies in the web: 0 on the screen's edge, 1 on the clear
   * ring round the great eye, and in between the share of the way from the
   * one to the other — the distance to the nearest edge against the distance
   * to the ring, so it runs straight from each edge in towards the eye.
   */
  deep(x, y) {
    const dx = x - this.cx, dy = y - this.cy;
    const edge = this.inset(x, y);
    const ring = Math.max(0, Math.sqrt(dx * dx + dy * dy) - this.ring);
    return edge + ring > 0 ? edge / (edge + ring) : 1;
  },

  /** The spacing wanted this deep: GAP_MIN at the edge, growing in a straight line to the ring. */
  gapAt(k) {
    return this.GAP_MIN + (this.gNext - this.GAP_MIN) * k;
  },

  /**
   * Eye e's radius this deep, at (x, y): a pixel at the edge, growing in a
   * straight line to the biggest by the ring, a little more or less for each
   * eye, and never so big as to reach into the clear ring.
   */
  sizeAt(k, e, x, y) {
    const dx = x - this.cx, dy = y - this.cy;
    const r = (this.EDGE + (this.rNext - this.EDGE) * k) * this.grain[e];
    return Math.max(0.3, Math.min(r, Math.sqrt(dx * dx + dy * dy) - this.clear));
  },

  /** How nervous an eye of this size is: not at all when small, wholly when it is the biggest. */
  nerves(r) {
    return Math.max(0, Math.min(1, (r - this.WHOLE) / (this.rNext - this.WHOLE)));
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

  /** How solid something is this deep: faint at the edge, where the web is the screen's own pixels. */
  solid(k) {
    return 0.28 + 0.72 * k * (1.6 - 0.6 * k);
  },

  /** One animation frame from Home, while the front page shows the eyes. */
  tick(dt) {
    if (!this.active || this.calm) return;
    this.time += dt;
    this._since = (this._since || 0) + dt;
    if (this._since < 1 / this.MAX_FPS) return;
    // A long pause (the tab was away) is taken as one short step, not a jump.
    const step = Math.min(0.1, this._since);
    this._since = 0;
    this.move(step);
    this.draw();
  },

  /**
   * One step of the web: every eye but the dots and the great eye carried
   * round its place, sized for where it now is, its threads chosen again and
   * faded towards what it chose, and its gaze moved on. `settle` puts every
   * thread straight where it belongs, for a first frame or a still picture.
   */
  move(dt, settle = false) {
    const { n, hx, hy, x, y, r, k, alpha, still, reach, gap, phx, phy, fx, fy, qx, qy,
            swell } = this;
    const calm = this.calm, t = this.time, TAU = Math.PI * 2;
    // Each current at this moment, so that where an eye is on it is a sum of
    // products rather than a sine an eye: sin(a + b) = sin a cos b + cos a sin b.
    const [s1, c1, s2, c2, s3, c3, s4, c4] = this.waves.flatMap(wave => [Math.sin(wave.beat * t), Math.cos(wave.beat * t)]);

    for (let e = 1; e < n; e++) {
      if (still[e]) continue;
      if (calm) { x[e] = hx[e]; y[e] = hy[e]; continue; }
      // Its own wander, at a pace that is greatest halfway in and slows
      // towards the edge and towards the great eye...
      const pace = 0.15 + 3.4 * k[e] * (1 - k[e]);
      const ax = (phx[e] += TAU * fx[e] * pace * dt), ay = (phy[e] += TAU * fy[e] * pace * dt);
      const ox = 0.65 * Math.sin(ax) + 0.35 * Math.sin(2.3 * ax + qx[e]);
      const oy = 0.65 * Math.sin(ay) + 0.35 * Math.sin(1.7 * ay + qy[e]);
      // ...and the currents it shares with its neighbours.
      const q = 8 * e;
      const sx = swell[q] * c1 + swell[q + 1] * s1 + swell[q + 2] * c2 + swell[q + 3] * s2;
      const sy = swell[q + 4] * c3 + swell[q + 5] * s3 + swell[q + 6] * c4 + swell[q + 7] * s4;
      x[e] = hx[e] + reach[e] * (0.25 * sx + 0.5 * ox);
      y[e] = hy[e] + reach[e] * (0.25 * sy + 0.5 * oy);
    }

    // How far apart each pair is now. Two eyes the wander has brought too
    // close are pushed apart on the way — room of their own, so that the web
    // never crowds whatever the currents do.
    const { pa, pb, pd, sig, want, pin } = this;
    const P = pa.length;
    for (let p = 0; p < P; p++) {
      const a = pa[p], b = pb[p];
      const dx = x[b] - x[a], dy = y[b] - y[a];
      let d = Math.sqrt(dx * dx + dy * dy);
      const need = 1.25 * (r[a] + r[b]) + 2;
      if (d < need && a !== 0 && !calm && d > 1e-6) {
        const push = (need - d) / d;
        const wa = still[a] ? 0 : still[b] ? 1 : 0.5, wb = still[b] ? 0 : still[a] ? 1 : 0.5;
        x[a] -= dx * push * wa; y[a] -= dy * push * wa;
        x[b] += dx * push * wb; y[b] += dy * push * wb;
        d = need;
      }
      pd[p] = d;
      // A dot's own threads were chosen once; they hold while the eye at the
      // other end stays within reach.
      want[p] = pin[p] && d < gap[a === 0 ? b : a] * 2.4 ? 1 : 0;
    }

    // Each eye's size, depth and strength, from where it is now.
    for (let e = 1; e < n; e++) {
      if (still[e]) continue;
      const kk = this.deep(x[e], y[e]);
      k[e] = kk;
      r[e] = this.sizeAt(kk, e, x[e], y[e]);
      alpha[e] = this.solid(kk);
    }

    // Each eye holds on to its nearest three, the great eye to its six. A
    // thread already there counts as a little nearer, so that two neighbours
    // at nearly the same distance do not trade the thread every frame.
    const best = this._best || (this._best = new Float64Array(12));
    const { adjStart, adj } = this;
    for (let e = 0; e < n; e++) {
      if (e !== 0 && still[e]) continue;
      const most = e === 0 ? 6 : 3;
      const far = e === 0 ? Infinity : gap[e] * 2.1;
      let held = 0;
      for (let q = adjStart[e]; q < adjStart[e + 1]; q++) {
        const p = adj[q];
        if (pd[p] > far) continue;
        const score = pd[p] * (sig[p] > 0.5 ? 0.88 : 1);
        // A short list kept in order: the nearest `most` so far.
        let at = held < most ? held++ : most;
        if (at === most && score >= best[2 * (most - 1)]) continue;
        if (at === most) at = most - 1;
        while (at > 0 && best[2 * (at - 1)] > score) {
          best[2 * at] = best[2 * (at - 1)];
          best[2 * at + 1] = best[2 * (at - 1) + 1];
          at--;
        }
        best[2 * at] = score;
        best[2 * at + 1] = p;
      }
      for (let h = 0; h < held; h++) {
        const p = best[2 * h + 1];
        // A thread not yet there grows out of the eye that reached for it.
        if (sig[p] === 0 && !want[p]) this.from[p] = pa[p] === e ? 0 : 1;
        want[p] = 1;
      }
    }
    const up = dt / this.KNIT, down = dt / this.FRAY;
    for (let p = 0; p < P; p++) {
      if (settle || calm) sig[p] = want[p];
      else sig[p] = want[p] ? Math.min(1, sig[p] + up) : Math.max(0, sig[p] - down);
    }

    // Where each whole eye looks: at the great eye, or away from it. The
    // bigger the eye, the more often it looks somewhere new, the quicker it
    // gets there and the further away it looks: the small ones gaze, the
    // big ones cannot settle.
    if (calm) return;
    const { gaze, gazeTo, gazeAt } = this;
    for (let e = 1; e < n; e++) {
      if (still[e] || r[e] < this.WHOLE) continue;
      const fear = this.nerves(r[e]);
      if (t >= gazeAt[e]) {
        const at = Math.random() < 0.8 - 0.3 * fear;
        gazeTo[e] = at ? (Math.random() - 0.5) * (0.9 - 0.7 * fear)
                       : (Math.random() < 0.5 ? -1 : 1) * (0.7 + 1.5 * fear + 0.5 * Math.random());
        gazeAt[e] = t + (2.8 - 2.45 * fear) * (0.4 + 1.2 * Math.random());
      }
      gaze[e] += (gazeTo[e] - gaze[e]) * (1 - Math.exp(-(2.5 + 30 * fear * fear) * dt));
    }
  },

  /** The inks of the eyes, as the theme has them now. */
  inks() {
    return { ball: Ink.of('eyeBall'), iris: Ink.of('eyeIris'), pupil: Ink.of('eyePupil'),
             rim: Ink.of('eyeRim'), link: Ink.of('eyeLink') };
  },

  /**
   * The part of the web that never moves — the dots at the edge and the
   * threads between them — drawn once, at full strength, into a picture kept
   * aside; draw() lays it down every frame, faded in as the web comes up.
   */
  paintStill() {
    if (!this.ctx || !this.n) return;
    const cache = this.cache || (this.cache = document.createElement('canvas'));
    cache.width = this.canvas.width;
    cache.height = this.canvas.height;
    const g = cache.getContext('2d');
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    g.clearRect(0, 0, this.width, this.height);
    const ink = this.inks();
    const { x, y, r, alpha, still, fixed } = this;
    const LEVELS = 8;
    const level = a => Math.min(LEVELS - 1, Math.floor(a * LEVELS));
    // Things of about one strength in one path, filled or stroked at once.
    const shade = (items, paint, ink) => {
      const buckets = Array.from({ length: LEVELS }, () => []);
      for (const item of items) buckets[level(item.alpha)].push(item);
      buckets.forEach((list, b) => {
        if (!list.length) return;
        g.globalAlpha = (b + 0.5) / LEVELS;
        g.beginPath();
        for (const item of list) paint(item);
        ink();
      });
    };

    // The threads between dots, straight.
    const threads = [];
    for (let f = 0; f < fixed.length; f += 2) {
      const a = fixed[f], b = fixed[f + 1];
      threads.push({ a, b, alpha: Math.min(alpha[a], alpha[b]) * 0.5 * this.grown(Math.min(r[a], r[b])) });
    }
    g.strokeStyle = ink.link;
    g.lineWidth = 0.8;
    shade(threads, ({ a, b }) => { g.moveTo(x[a], y[a]); g.lineTo(x[b], y[b]); }, () => g.stroke());

    // The dots: the screen's own pixels, in the iris's colour.
    const dots = [];
    for (let e = 1; e < this.n; e++) if (still[e]) dots.push({ e, alpha: Math.min(1, alpha[e] * 1.6) });
    g.fillStyle = ink.iris;
    shade(dots, ({ e }) => {
      const d = Math.max(1, Math.round(r[e] * 2));
      g.rect(Math.round(x[e] - d / 2), Math.round(y[e] - d / 2), d, d);
    }, () => g.fill());
    g.globalAlpha = 1;
  },

  draw() {
    const ctx = this.ctx;
    if (!ctx || !this.n) return;
    const calm = this.calm;
    // A still picture is the web a few seconds in, fully grown.
    const t = calm ? 6 : this.time;
    const rise = calm ? 1 : Math.min(1, t / this.FADE_IN);
    const fade = rise * rise * (3 - 2 * rise);
    const ink = this.inks();
    const { n, x, y, r, alpha, still, pa, pb, pd, sig, w1, w2, from } = this;
    const LEVELS = 8, TAU = Math.PI * 2;

    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    if (this.cache) {
      ctx.globalAlpha = fade;
      ctx.drawImage(this.cache, 0, 0, this.width, this.height);
    }

    // ---- the threads that move, in a few strengths so each is one stroke call ----
    // A thread coming or going is part grown: it reaches out of the eye that
    // chose it towards the other, brightening as it goes, and draws back
    // into that eye as it frays.
    const [threads, dots, small, whole] = this.bins;
    for (const bin of this.bins) for (const list of bin) list.length = 0;
    this.large.length = 0;
    for (let p = 0; p < pa.length; p++) {
      const s = sig[p];
      if (s <= 0) continue;
      const a = pa[p], b = pb[p];
      const strength = Math.min(alpha[a], alpha[b]) * 0.5 * this.grown(Math.min(r[a], r[b]))
                     * Math.min(1, s * 2) * fade;
      if (strength < 0.016) continue;
      // The tremble: two points along the thread, each pushed sideways by a
      // little more or less every frame and drawn back towards the line. A
      // short thread would tremble by less than a pixel, so it is drawn straight.
      if (!calm && pd[p] > 12) {
        const amp = Math.min(3, pd[p] * 0.04);
        w1[p] = Math.max(-amp, Math.min(amp, w1[p] * 0.9 + (Math.random() - 0.5) * amp * 0.5));
        w2[p] = Math.max(-amp, Math.min(amp, w2[p] * 0.9 + (Math.random() - 0.5) * amp * 0.5));
      }
      threads[Math.min(LEVELS - 1, Math.floor(strength / 0.5 * LEVELS))].push(p);
    }
    ctx.strokeStyle = ink.link;
    ctx.lineWidth = 1;
    threads.forEach((list, level) => {
      if (!list.length) return;
      ctx.globalAlpha = ((level + 0.5) / LEVELS) * 0.5;
      ctx.beginPath();
      for (const p of list) {
        const a = from[p] ? pb[p] : pa[p], b = from[p] ? pa[p] : pb[p];
        const s = sig[p], grow = s >= 1 ? 1 : s * s * (3 - 2 * s);
        const px = x[a], py = y[a], dx = (x[b] - px) * grow, dy = (y[b] - py) * grow;
        ctx.moveTo(px, py);
        if (pd[p] > 12) {
          const length = pd[p] || 1;
          const nx = -dy / length, ny = dx / length;
          ctx.lineTo(px + dx / 3 + nx * w1[p], py + dy / 3 + ny * w1[p]);
          ctx.lineTo(px + dx * 2 / 3 + nx * w2[p], py + dy * 2 / 3 + ny * w2[p]);
        }
        ctx.lineTo(px + dx, py + dy);
      }
      ctx.stroke();
    });

    // ---- the eyes that move, sorted by size and strength ----
    const { ix, iy, open, gaze, blink, blinkAt } = this;
    const cx = this.cx, cy = this.cy;
    for (let e = 1; e < n; e++) {
      if (still[e]) continue;
      const a = alpha[e] * fade;
      if (a < 0.02) continue;
      const re = r[e];
      if (re < this.DOT) { dots[Math.min(LEVELS - 1, Math.floor(Math.min(1, a * 1.6) * LEVELS))].push(e); continue; }
      const level = Math.min(LEVELS - 1, Math.floor(a * LEVELS));
      if (re < this.WHOLE) { small[level].push(e); continue; }
      // A blink is a quick close and open, a few times a minute.
      const phase = (t + blinkAt[e]) % blink[e];
      const shut = calm ? 1 : (phase < 0.16 ? Math.abs(Math.cos(Math.PI * phase / 0.16)) : 1);
      open[e] = Math.max(0.06, shut);
      // Towards the great eye, or off it, as move() has it; a nervous eye's
      // pupil shakes as well, once it is big enough for the shaking to show.
      const fear = calm ? 0 : this.nerves(re);
      const shake = fear * re > 3 ? fear : 0;
      const look = Math.atan2(cy - y[e], cx - x[e]) + (calm ? 0 : gaze[e])
                 + (shake ? (Math.random() - 0.5) * shake * 0.3 : 0);
      const off = (0.3 + 0.06 * fear) * re;
      ix[e] = x[e] + Math.cos(look) * off + (shake ? (Math.random() - 0.5) * shake * re * 0.05 : 0);
      iy[e] = y[e] + Math.sin(look) * off * open[e] + (shake ? (Math.random() - 0.5) * shake * re * 0.05 : 0);
      if (re < this.LARGE) whole[level].push(e);
      else this.large.push(e);
    }

    // The dots among them: pixels, in the iris's colour.
    ctx.fillStyle = ink.iris;
    dots.forEach((list, level) => {
      if (!list.length) return;
      ctx.globalAlpha = (level + 0.5) / LEVELS;
      ctx.beginPath();
      for (const e of list) {
        const d = Math.max(1, Math.round(r[e] * 2));
        ctx.rect(Math.round(x[e] - d / 2), Math.round(y[e] - d / 2), d, d);
      }
      ctx.fill();
    });

    // The small eyes: a ball and a pupil, nothing more at this size.
    small.forEach((list, level) => {
      if (!list.length) return;
      ctx.globalAlpha = (level + 0.5) / LEVELS;
      ctx.fillStyle = ink.ball;
      ctx.beginPath();
      for (const e of list) { ctx.moveTo(x[e] + r[e], y[e]); ctx.arc(x[e], y[e], r[e], 0, TAU); }
      ctx.fill();
      ctx.fillStyle = ink.pupil;
      ctx.beginPath();
      for (const e of list) { ctx.moveTo(x[e] + r[e] * 0.45, y[e]); ctx.arc(x[e], y[e], r[e] * 0.45, 0, TAU); }
      ctx.fill();
    });

    // The whole eyes of a middling size, a layer at a time: every ball, every
    // rim, every iris, every pupil, every glint. They rarely overlap, so the
    // order between them does not show, and it is a few calls, not thousands.
    // An eye between blinks is round, and a circle is cheaper to lay down
    // than an ellipse.
    const oval = (ox, oy, radius, squash) => {
      ctx.moveTo(ox + radius, oy);
      if (squash >= 1) ctx.arc(ox, oy, radius, 0, TAU);
      else ctx.ellipse(ox, oy, radius, radius * squash, 0, 0, TAU);
    };
    whole.forEach((list, level) => {
      if (!list.length) return;
      const a = (level + 0.5) / LEVELS;
      ctx.globalAlpha = a;
      ctx.fillStyle = ink.ball;
      ctx.beginPath();
      for (const e of list) oval(x[e], y[e], r[e], open[e]);
      ctx.fill();
      ctx.strokeStyle = ink.rim;
      ctx.lineWidth = 0.8;
      ctx.globalAlpha = a * 0.55;
      ctx.stroke();
      ctx.globalAlpha = a;
      ctx.fillStyle = ink.iris;
      ctx.beginPath();
      for (const e of list) if (open[e] >= 0.25) oval(ix[e], iy[e], r[e] * 0.6, open[e]);
      ctx.fill();
      ctx.fillStyle = ink.pupil;
      ctx.beginPath();
      for (const e of list) if (open[e] >= 0.25) oval(ix[e], iy[e], r[e] * 0.31, open[e]);
      ctx.fill();
      // The glint, on eyes big enough for it to be more than a speck.
      ctx.fillStyle = ink.ball;
      ctx.globalAlpha = a * 0.85;
      ctx.beginPath();
      for (const e of list) {
        if (open[e] < 0.25 || r[e] < 6) continue;
        oval(x[e] - r[e] * 0.26, y[e] - r[e] * 0.28 * open[e], r[e] * 0.09, 1);
      }
      ctx.fill();
    });

    // The big ones, each on its own.
    for (const e of this.large) {
      const a = alpha[e] * fade;
      this.eye(ctx, x[e], y[e], r[e], open[e], a, ink, { jx: ix[e] - x[e], jy: iy[e] - y[e] });
    }

    // ---- the great eye, looking at you ----
    const m = this.big, px = cx, py = cy;
    // A tremor every frame, and every second or so a dart to a new resting
    // place: an eye that is trying to hold still and cannot.
    let jx = 0, jy = 0;
    if (!calm) {
      if (t >= this.dart.next) {
        const a = Math.random() * TAU;
        const d = Math.random() * m * 0.05;
        this.dart = { x: Math.cos(a) * d, y: Math.sin(a) * d, next: t + 0.6 + Math.random() * 1.3 };
      }
      jx = this.dart.x + (Math.random() - 0.5) * m * 0.04;
      jy = this.dart.y + (Math.random() - 0.5) * m * 0.04;
    }
    const breathe = calm ? 1 : 1 + 0.05 * Math.sin(t * 1.7);

    // A faint light behind it, so it sits in the web rather than on it.
    ctx.globalAlpha = 0.22 * fade;
    const glow = ctx.createRadialGradient(px, py, m * 0.8, px, py, m * 2.4);
    glow.addColorStop(0, ink.iris);
    glow.addColorStop(1, Ink.alpha(ink.iris, 0));
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(px, py, m * 2.4, 0, TAU); ctx.fill();

    this.eye(ctx, px, py, m, 1, fade, ink, { jx, jy, breathe, spokes: true });
    ctx.globalAlpha = 1;
  },

  /**
   * One eye: the ball, its rim, the iris and the pupil, the iris and pupil
   * moved off the middle by (`jx`, `jy`) — where a glance has put them, or
   * the great eye's tremor; `open` closes it towards a line. The great eye
   * adds the iris's spokes and a pupil that swells and shrinks (`breathe`).
   */
  eye(ctx, x, y, r, open, alpha, ink, { jx = 0, jy = 0, breathe = 1, spokes = false } = {}) {
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

    const ix = x + jx, iy = y + jy;
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
