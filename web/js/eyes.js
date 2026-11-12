/*
 * The front page's other backdrop: a network of eyes, grown out of the
 * screen itself, round one great eye that looks back at you.
 *
 * The mark is six agents, each one an eye, the middle one joined to the rest;
 * this is the mark come alive. The great eye stands where the mark stands,
 * over the title, and stares straight out: its pupil sits in the middle of
 * the ball and will not keep still, trembling a little every frame and now
 * and then darting. The eye itself holds its place, inside a small clear ring
 * that no other eye enters. None of the others looks at you; they look at the
 * great eye — the small ones calmly, the big ones nervously, in quick glances
 * at it and away. The threads between them are straight, and tremble a little
 * at random, as a wire does.
 *
 * The screen is an old tube monitor, and its pixels are eyes. Its glass is a
 * rounded square, bulging a little, so the picture bends in towards the
 * corners; round it runs a dark rim. Along the edge the eyes sit still in a
 * raster, tiny and in rows, with a line along every row, a wash and a grain
 * in the eyes' own colour and the rim's shadow on the glass — strongest at
 * the edge and gone well before the middle. None of it moves: the edge is a
 * screen standing still. Inwards the raster loosens and the eyes come free,
 * take threads, and grow in a straight line all the way to the clear ring. An
 * eye's size is where it is, so one that travels in grows, and one that
 * travels out shrinks until it sinks back among the pixels. The words lie
 * over the web and do not part it; the title carries its own shadow
 * (style.css).
 *
 * Every free eye travels. Slow currents, shared by neighbours, carry them
 * round the screen, each with a wander of its own on top, fastest halfway in
 * and slower by the edge and by the great eye. Each keeps a little room round
 * itself, so they never crowd or leave holes, and the spacing stays what the
 * place asks for. As they go each holds on to its nearest three, so threads
 * are always being made and let go: a new one grows out of the eye that
 * reached for it, an old one draws back into it as it frays.
 *
 * Cheap on purpose. The tube — rim, glass and raster — is drawn once into a
 * picture kept aside and laid down every frame. The free eyes, a thousand or
 * so on a large screen, live in typed arrays and in a grid rebuilt every
 * frame; each works out the stream and looks round itself every other frame,
 * and they are drawn with their threads in a few batches, at most thirty
 * times a second, and only from Home.tick — so only while the front page is
 * on screen. For a reader who asks for less motion it is one still picture,
 * drawn again only when the page changes size or colour.
 */
const Eyes = {
  /** Draws a second, at most: a backdrop, like the recorded world. */
  MAX_FPS: 30,

  /** Seconds the web takes to come up out of the glass on arrival. */
  FADE_IN: 1.8,

  /** How many rows deep the raster of pixel-eyes runs in from the edge. */
  ROWS: 9,

  /** Below this radius an eye is a dot; from it, a small eye; from WHOLE, a whole eye. */
  DOT: 1.6,

  /** From this radius a dot takes threads, faint at first, as strong as an eye's by WHOLE. */
  THREAD: 1,

  /** From this radius an eye is whole: a white ball round its iris; it looks about and blinks. */
  WHOLE: 3.5,

  /** From this radius an eye is drawn on its own, its rim as heavy as its size. */
  LARGE: 13,

  /** The clear ring round the great eye, in radii of the great eye. */
  CLEAR: 1.95,

  /** The eyes just outside the clear ring, the biggest after the great eye, in its radii. */
  NEXT: 0.45,

  /** The spacing there, in radii of the eyes there. */
  ROOM: 5.2,

  /** How fast the free eyes travel, in spacings a second at the pace's top. */
  SPEED: 0.75,

  /** Seconds a new thread takes to grow, and an old one to fade away. */
  KNIT: 0.6,
  FRAY: 0.9,

  /** Threads an eye keeps track of: its nearest three, and room for those still fading. */
  SLOTS: 8,

  /** How far the picture bends in towards the corners of the bulging glass, as a share of the half-width. */
  BULGE: 0.04,

  canvas: null,
  host: null,
  anchor: null,
  active: false,
  time: 0,
  n: 0,
  cache: null,
  tint: null,

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
   * The web for a screen of this size. First the free eyes' places: the
   * screen is cut into square cells, each cut in four wherever it is wider
   * than the spacing wanted at its middle — a little more or less at random,
   * so the steps between sizes do not show as rings — and each cell offers one
   * place; the places are taken biggest first, and one too near a place
   * already taken is dropped, so the eyes start out evenly spread. Then the
   * raster round the edge, a pixel-eye at every step of its grid. Seeded, so
   * the same screen grows the same web.
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

    // The tube's face: just inside the stage, leaving a rim at the middle of
    // each side and more at the rounded corners.
    this.rim = Math.max(5, Math.min(14, Math.round(Math.min(width, height) * 0.012)));
    this.ax = width / 2 - this.rim;
    this.ay = height / 2 - this.rim;

    // Where the great eye stands on the glass: where the mark is, or the
    // middle. Its place in the picture is that point with the bulge undone.
    const box = host.getBoundingClientRect();
    const mark = this.anchor ? this.anchor.getBoundingClientRect() : null;
    const showX = mark && mark.width ? mark.left + mark.width / 2 - box.left : width / 2;
    const showY = mark && mark.height ? mark.top + mark.height / 2 - box.top : height * 0.55;
    const big = Math.max(34, Math.min(96, (mark && mark.width ? mark.width : 140) * 0.42));
    this.unwarp(showX, showY);
    const cx = this.atX, cy = this.atY;
    this.cx = cx;
    this.cy = cy;
    this.big = big;

    // The clear ring, and just outside it the biggest of the other eyes, their
    // middles a little further out so that their balls only touch it.
    this.clear = big * this.CLEAR;
    this.rNext = big * this.NEXT;
    this.ring = this.clear + this.rNext;
    this.gNext = this.rNext * this.ROOM;

    // The raster: a pixel-eye every `pitch` pixels, ROWS rows deep. The free
    // eyes begin just inside it, a little looser and a little bigger than its
    // pixels, and grow in a straight line from there to the ring.
    const pitch = Math.max(4, Math.min(7, Math.round(Math.min(width, height) / 170)));
    this.pitch = pitch;
    this.depth = pitch * this.ROWS;
    this.gEdge = pitch * 1.5;
    this.rEdge = pitch * 0.36;

    const rand = seededRandom(4111);
    const TAU = Math.PI * 2;
    const ring = this.ring, depth = this.depth;
    const free = (px, py) => this.inset(px, py) >= depth + this.gEdge * 0.3 && Math.hypot(px - cx, py - cy) >= ring;

    // ---- the free eyes' places on offer, one a cell ----
    const offer = [];
    const side = 2 ** Math.ceil(Math.log2(Math.max(width, height)));
    const cells = [[0, 0, side]];
    while (cells.length) {
      const [x, y, s] = cells.pop();
      if (x >= width || y >= height) continue;
      const mx = Math.min(x + s / 2, width), my = Math.min(y + s / 2, height);
      // A cell wholly inside the raster has nothing to offer.
      if (this.inset(mx, my) + s * 0.71 < depth) continue;
      const want = this.gapAt(this.deep(mx, my));
      if (s > this.gEdge * 0.5 && s > want * (0.7 + 0.6 * rand())) {
        const h = s / 2;
        cells.push([x, y, h], [x + h, y, h], [x, y + h, h], [x + h, y + h, h]);
        continue;
      }
      // A cell the raster or the clear ring cuts tries a few times, a little
      // wider each time, for the part of it that is free.
      for (let attempt = 0; attempt < 6; attempt++) {
        const spread = 0.6 + 0.06 * attempt;
        const px = x + s * (0.5 + spread * (rand() - 0.5)), py = y + s * (0.5 + spread * (rand() - 0.5));
        if (px >= width || py >= height) break;
        if (!free(px, py)) continue;
        offer.push(px, py);
        break;
      }
    }

    // ---- taken biggest first, none too near another ----
    const offered = offer.length / 2;
    const wants = new Float32Array(offered);
    for (let i = 0; i < offered; i++) wants[i] = this.gapAt(this.deep(offer[2 * i], offer[2 * i + 1]));
    const order = Array.from({ length: offered }, (_, i) => i).sort((a, b) => wants[b] - wants[a]);
    const CELL = 16, cw = Math.ceil(width / CELL) + 1, ch = Math.ceil(height / CELL) + 1;
    const first = new Int32Array(cw * ch).fill(-1);
    const after = new Int32Array(offered);
    const taken = [];
    for (const i of order) {
      const px = offer[2 * i], py = offer[2 * i + 1], want = wants[i];
      const span = Math.ceil((want * 1.6 + 2) / CELL);
      const gx = Math.floor(px / CELL), gy = Math.floor(py / CELL);
      let room = true;
      for (let b = Math.max(0, gy - span); room && b <= Math.min(ch - 1, gy + span); b++) {
        for (let a = Math.max(0, gx - span); room && a <= Math.min(cw - 1, gx + span); a++) {
          for (let j = first[b * cw + a]; j >= 0; j = after[j]) {
            if (Math.hypot(offer[2 * j] - px, offer[2 * j + 1] - py) < 0.4 * (want + wants[j])) { room = false; break; }
          }
        }
      }
      if (!room) continue;
      after[i] = first[gy * cw + gx];
      first[gy * cw + gx] = i;
      taken.push(i);
    }

    // ---- the raster: the screen's own pixels ----
    // In rows at the very edge; looser further in, where the eyes come free.
    const raster = [];
    for (let j = 0; (j + 0.5) * pitch < height; j++) {
      for (let i = 0; (i + 0.5) * pitch < width; i++) {
        const px = (i + 0.5) * pitch, py = (j + 0.5) * pitch;
        const e = this.inset(px, py);
        // Out to a little beyond the face's outline, under the rim, so the
        // glass is full to its edge.
        if (e >= depth || e < -pitch || Math.hypot(px - cx, py - cy) < ring) continue;
        const loose = (Math.max(0, e) / depth) ** 2;
        raster.push(px + (rand() - 0.5) * 0.7 * pitch * loose, py + (rand() - 0.5) * 0.7 * pitch * loose, loose);
      }
    }

    // ---- the eyes: the great eye, then the free ones, then the raster ----
    const F = taken.length, R = raster.length / 3, n = 1 + F + R, S = this.SLOTS;
    const all = () => new Float32Array(n), own = () => new Float32Array(F + 1);
    Object.assign(this, {
      n, F,
      x: all(), y: all(),               // where it is in the picture
      sx: all(), sy: all(),             // where that shows on the bulging glass
      r: all(), k: all(), alpha: all(), // its size, how deep it is, how solid
      gap: all(),                       // the spacing where it is
      grain: all(),                     // its own size beside its neighbours'
      tie: new Uint8Array(n),           // a thread can be tied to it
      vx: own(), vy: own(),             // how it is travelling
      wx: own(), wy: own(),             // its own wander
      tx: own(), ty: own(), roam: own(), // where the stream pulls it, and how far its wander reaches
      gaze: own(), gazeTo: own(), gazeAt: new Float64Array(F + 1),
      blink: own(), blinkAt: own(),
      ix: own(), iy: own(), open: own(), // where the pupil is and how open the eye, this frame
      // Its threads, held by the eye that reached for them: who is at the
      // other end, how far grown, how it trembles.
      mate: new Int32Array((F + 1) * S).fill(-1),
      sig: new Float32Array((F + 1) * S),
      w1: new Float32Array((F + 1) * S),
      w2: new Float32Array((F + 1) * S),
      keep: new Uint8Array((F + 1) * S)
    });

    this.x[0] = cx;
    this.y[0] = cy;
    this.sx[0] = showX;
    this.sy[0] = showY;
    this.r[0] = big;
    this.k[0] = 1;
    this.alpha[0] = 1;
    this.gap[0] = this.gNext;
    this.tie[0] = 1;
    taken.forEach((i, at) => {
      const e = at + 1;
      this.x[e] = offer[2 * i];
      this.y[e] = offer[2 * i + 1];
      this.grain[e] = 0.8 + 0.4 * rand();
      this.tie[e] = 1;
      this.wx[e] = (rand() - 0.5) * 0.8;
      this.wy[e] = (rand() - 0.5) * 0.8;
      // When it blinks: every few seconds, a different few for each.
      this.blink[e] = 2.5 + 7 * rand();
      this.blinkAt[e] = rand() * 9;
      this.gazeAt[e] = rand() * 2;
    });
    this.size();
    for (let q = 0; q < R; q++) {
      const e = 1 + F + q, loose = raster[3 * q + 2];
      this.x[e] = raster[3 * q];
      this.y[e] = raster[3 * q + 1];
      this.warp(this.x[e], this.y[e]);
      this.sx[e] = this.atX;
      this.sy[e] = this.atY;
      this.r[e] = pitch * (0.27 + 0.1 * loose) * (0.9 + 0.2 * rand());
      this.alpha[e] = 0.75 - 0.4 * loose;
      this.gap[e] = pitch;
      this.k[e] = 0;
      // Only its innermost row reaches out to the free eyes.
      this.tie[e] = loose > 0.76 ? 1 : 0;
    }

    // The threads among the loosened pixels near the inside of the raster,
    // each to its nearest two: drawn once, with the raster.
    const fixed = [];
    const near = new Map(), ROW = 1 << 16;
    for (let e = 1 + F; e < n; e++) {
      if (raster[3 * (e - 1 - F) + 2] < 0.4) continue;
      const key = Math.floor(this.x[e] / (2 * pitch)) * ROW + Math.floor(this.y[e] / (2 * pitch));
      if (!near.has(key)) near.set(key, []);
      near.get(key).push(e);
    }
    for (const [key, list] of near) {
      const gx = Math.floor(key / ROW), gy = key - gx * ROW;
      for (const e of list) {
        const found = [];
        for (let b = gy - 1; b <= gy + 1; b++) {
          for (let a = gx - 1; a <= gx + 1; a++) {
            for (const j of near.get(a * ROW + b) || []) {
              if (j <= e) continue;
              const d = Math.hypot(this.x[j] - this.x[e], this.y[j] - this.y[e]);
              if (d < 1.75 * pitch) found.push([j, d]);
            }
          }
        }
        found.sort((p, q) => p[1] - q[1]);
        for (const [j] of found.slice(0, 2)) fixed.push(e, j);
      }
    }
    this.fixed = Int32Array.from(fixed);

    // The currents: a few waves about a screen long, which together make the
    // stream the free eyes flow along (move()); each sized so that their slope
    // together is about one. They shift only slowly, over a minute or two, so
    // an eye is carried a long way round an eddy before the eddy moves on.
    const length = Math.min(width, height), WAVES = 5;
    this.waves = Array.from({ length: WAVES }, () => {
      const a = rand() * TAU, kk = TAU / (length * (0.45 + 0.5 * rand()));
      return { kx: Math.cos(a) * kk, ky: Math.sin(a) * kk, at: rand() * TAU,
               beat: TAU * (0.006 + 0.01 * rand()) * (rand() < 0.5 ? -1 : 1),
               amp: 1 / (kk * Math.sqrt(WAVES / 2)) };
    });

    // The grid everyone with a thread to offer is filed in, every frame.
    this.cell = 4 * pitch;
    this.gw = Math.ceil(width / this.cell) + 1;
    this.gh = Math.ceil(height / this.cell) + 1;
    this.head = new Int32Array(this.gw * this.gh);
    this.next = new Int32Array(n);
    this.ties = [];
    for (let e = 1 + F; e < n; e++) if (this.tie[e]) this.ties.push(e);

    // Kept between frames, so drawing makes no garbage.
    this.bins = Array.from({ length: 4 }, () => Array.from({ length: 8 }, () => []));
    this.large = [];
    this.dart = { x: 0, y: 0, next: 0 };
    this.from = new Uint8Array((F + 1) * S);
    this.shown = new Float32Array((F + 1) * S);

    // The first frame shows each eye already holding its threads.
    this.move(0, true);
    this.paintStill();
  },

  /**
   * How deep a point lies in the web: 0 at the inside of the raster, 1 on the
   * clear ring round the great eye, and in between the share of the way from
   * the one to the other — the distance from the raster against the distance
   * to the ring, so it runs straight from each edge in towards the eye.
   */
  deep(x, y) {
    const dx = x - this.cx, dy = y - this.cy;
    const edge = Math.max(0, this.inset(x, y) - this.depth);
    const ring = Math.max(0, Math.sqrt(dx * dx + dy * dy) - this.ring);
    return edge + ring > 0 ? edge / (edge + ring) : 1;
  },

  /** The spacing wanted this deep: a little looser than the raster at its inside, growing in a straight line to the ring. */
  gapAt(k) {
    return this.gEdge + (this.gNext - this.gEdge) * k;
  },

  /**
   * Eye e's radius this deep, at (x, y): a raster pixel's at the inside of the
   * raster, growing in a straight line to the biggest by the ring, a little
   * more or less for each eye, and never so big as to reach into the clear ring.
   */
  sizeAt(k, e, x, y) {
    const dx = x - this.cx, dy = y - this.cy;
    const r = (this.rEdge + (this.rNext - this.rEdge) * k) * this.grain[e];
    return Math.max(0.3, Math.min(r, Math.sqrt(dx * dx + dy * dy) - this.clear));
  },

  /** Every free eye's depth, spacing, size and strength, from where it now is. */
  size() {
    const { F, x, y, k, gap, r, alpha } = this;
    for (let e = 1; e <= F; e++) {
      const kk = this.deep(x[e], y[e]);
      k[e] = kk;
      gap[e] = this.gapAt(kk);
      r[e] = this.sizeAt(kk, e, x[e], y[e]);
      alpha[e] = this.solid(kk);
    }
  },

  /** How nervous an eye of this size is: not at all when small, wholly when it is the biggest. */
  nerves(r) {
    return Math.max(0, Math.min(1, (r - this.WHOLE) / (this.rNext - this.WHOLE)));
  },

  /**
   * How strong a thread is for the size of the smaller eye it joins: next to
   * nothing between the finest dots, full between whole eyes, so the web
   * condenses out of the pixels rather than starting at a line.
   */
  grown(r) {
    const k = Math.max(0, Math.min(1, (r - this.THREAD) / (this.WHOLE - this.THREAD)));
    return 0.08 + 0.92 * k * k * (3 - 2 * k);
  },

  /**
   * How far a point of the picture lies inside the tube's face, near enough:
   * how far short of the outline it is, over how fast that changes. The face
   * is |X|⁶ + |Y|⁶ = 1 on the stage's half-sizes less the rim — a square with
   * well rounded corners. Leaves the way inwards in `inX`, `inY`.
   */
  face(x, y) {
    const X = (x - this.width / 2) / this.ax, Y = (y - this.height / 2) / this.ay;
    const X2 = X * X, Y2 = Y * Y, X4 = X2 * X2, Y4 = Y2 * Y2, sum = X4 * X2 + Y4 * Y2;
    if (sum < 1e-12) { this.inX = 0; this.inY = 0; return Math.min(this.ax, this.ay); }
    const rho = Math.cbrt(Math.sqrt(sum)), r5 = rho * rho * rho * rho * rho;
    const gx = X4 * X / (this.ax * r5), gy = Y4 * Y / (this.ay * r5);
    const slope = Math.sqrt(gx * gx + gy * gy) || 1e-9;
    this.inX = -gx / slope;
    this.inY = -gy / slope;
    return (1 - rho) / slope;
  },

  /** How far a point lies inside the tube's face; negative outside it. */
  inset(x, y) {
    return this.face(x, y);
  },

  /**
   * Where a point of the picture shows on the bulging glass, left in `atX`,
   * `atY`: pulled in towards the corners, the more the further out.
   */
  warp(x, y) {
    const hw = this.width / 2, hh = this.height / 2, b = this.BULGE;
    const X = (x - hw) / hw, Y = (y - hh) / hh;
    this.atX = hw + X * (1 - b * Y * Y) * hw;
    this.atY = hh + Y * (1 - b * X * X) * hh;
  },

  /** The point of the picture that shows at (x, y) on the glass, left in `atX`, `atY`: the bulge undone, by a few steps of guessing. */
  unwarp(x, y) {
    const hw = this.width / 2, hh = this.height / 2, b = this.BULGE;
    const Xs = (x - hw) / hw, Ys = (y - hh) / hh;
    let X = Xs, Y = Ys;
    for (let i = 0; i < 3; i++) {
      X = Xs / (1 - b * Y * Y);
      Y = Ys / (1 - b * X * X);
    }
    this.atX = hw + X * hw;
    this.atY = hh + Y * hh;
  },

  /** How solid an eye is this deep: faint where it has just come off the raster. */
  solid(k) {
    return 0.28 + 0.72 * k * (1.6 - 0.6 * k);
  },

  /** How strongly the glass shows at a point this far inside the face: fully at the edge, gone a few raster depths in. */
  glass(inset) {
    const lo = this.depth * 0.35, hi = this.depth * 2.6;
    const u = Math.max(0, Math.min(1, (inset - lo) / (hi - lo)));
    return 1 - u * u * (3 - 2 * u);
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
   * Where the stream pulls free eye e at time t, into tx and ty, and how far
   * its own wander reaches, into roam. The stream is a field of slow waves N,
   * weighted by h, which is nothing at the raster and at the clear ring and
   * greatest a little way in from the raster; the eyes flow along its
   * contours at the spacing squared times its slope. Flowing so, they can
   * never gather anywhere or thin out — there stay as many eyes to a
   * spacing's square as there were — so the sizes keep their order however
   * far the eyes go; and nothing flows into the raster or the ring, only along
   * them.
   */
  stream(e, t) {
    const px = this.x[e], py = this.y[e];
    // How deep it is, k, and which way is deeper.
    const edge = this.face(px, py), ex = this.inX, ey = this.inY;
    const ox = px - this.cx, oy = py - this.cy, dist = Math.sqrt(ox * ox + oy * oy) || 1;
    const a = Math.max(0, edge - this.depth), b = Math.max(0, dist - this.ring), sum = a + b || 1;
    const k = a / sum, kx = (b * ex - a * ox / dist) / (sum * sum), ky = (b * ey - a * oy / dist) / (sum * sum);
    // The weight h, the pace over the spacing: the pace is nothing at either
    // end and rises and falls between, flat at the ring, so the eyes beside
    // the great eye barely stir.
    const g0 = this.gEdge, dg = this.gNext - this.gEdge, gp = g0 + dg * k, sn = Math.sin(Math.PI * k);
    const pace = sn * (1 - k), paced = Math.PI * Math.cos(Math.PI * k) * (1 - k) - sn;
    const h = pace / gp, hd = (paced * gp - pace * dg) / (gp * gp);
    let nv = 0, nx = 0, ny = 0;
    for (const w of this.waves) {
      const th = w.kx * px + w.ky * py + w.at + w.beat * t;
      const c = Math.cos(th) * w.amp;
      nv += Math.sin(th) * w.amp;
      nx += w.kx * c;
      ny += w.ky * c;
    }
    // The slope of the stream h·N, turned a quarter.
    const sx = hd * kx * nv + h * nx, sy = hd * ky * nv + h * ny;
    const flow = this.SPEED * gp * gp;
    this.tx[e] = flow * sy;
    this.ty[e] = -flow * sx;
    this.roam[e] = 0.3 * gp * pace + 1e-6;
  },

  /** Which of eye p's slots holds a thread to eye q, or -1. */
  slotOf(p, q) {
    const S = this.SLOTS, mate = this.mate;
    for (let s = p * S; s < p * S + S; s++) if (mate[s] === q) return s;
    return -1;
  },

  /**
   * One step of the web: every free eye carried on by the currents and its
   * own wander, given room by its neighbours, kept off the raster and out of
   * the clear ring, and sized for where it now is; its threads chosen again
   * and grown or frayed towards what it chose; its gaze moved on. `settle`
   * puts every thread straight where it belongs and moves nothing, for a
   * first frame or a still picture.
   */
  move(dt, settle = false) {
    const { F, x, y, gap, k, vx, vy, wx, wy, mate, sig, keep } = this;
    const calm = this.calm, hold = calm || settle, t = this.time, S = this.SLOTS;

    // ---- where each free eye is heading ----
    // Each is carried along by the stream (stream()) and, on top, by a wander
    // of its own that drifts at random and is drawn back towards nothing. The
    // stream changes slowly along an eye's way, so each works out its pull
    // every other frame, in turn with half the others, and steers towards it
    // every frame.
    const turn = (this._turn = (this._turn || 0) + 1) & 1;
    const { tx, ty, roam } = this;
    if (!hold) {
      const ease = Math.min(1, dt * 3), drift = Math.sqrt(dt) * 0.9, back = dt * 0.7;
      for (let e = 1; e <= F; e++) {
        wx[e] += (Math.random() - 0.5) * drift - wx[e] * back;
        wy[e] += (Math.random() - 0.5) * drift - wy[e] * back;
        if ((e & 1) === turn || roam[e] === 0) this.stream(e, t);
        vx[e] += (tx[e] + wx[e] * roam[e] - vx[e]) * ease;
        vy[e] += (ty[e] + wy[e] * roam[e] - vy[e]) * ease;
      }
    }

    // ---- everyone a thread can be tied to, filed by place ----
    const CELL = this.cell, gw = this.gw, gh = this.gh, head = this.head, next = this.next;
    head.fill(-1);
    const file = e => {
      const c = Math.floor(y[e] / CELL) * gw + Math.floor(x[e] / CELL);
      next[e] = head[c];
      head[c] = e;
    };
    for (let e = 0; e <= F; e++) file(e);
    for (const e of this.ties) file(e);

    // ---- room, and the nearest three ----
    // Each free eye looks round itself: anyone nearer than the spacing there
    // pushes it away a little, so the web never crowds; and it notes its
    // nearest three to hold on to — keeping one it holds already over a new
    // one that is only a little nearer, so threads do not flicker between two.
    // Half the free eyes look each frame, in turn, and push twice as hard for
    // it; the great eye looks every frame, and only holds on, to its nearest six.
    const best = this._best || (this._best = new Float64Array(14));
    const up = settle ? 1 : dt / this.KNIT, down = settle ? 1 : dt / this.FRAY;
    const r = this.r;
    for (let e = 0; e <= F; e++) {
      const base = e * S;
      if (e === 0 || hold || (e & 1) === turn) {
        const most = e === 0 ? 6 : 3;
        const far = e === 0 ? this.ring + 2.2 * this.gNext : 1.5 * gap[e];
        const span = Math.ceil(far / CELL);
        const gx = Math.floor(x[e] / CELL), gy = Math.floor(y[e] / CELL);
        let held = 0, px = 0, py = 0;
        for (let b = Math.max(0, gy - span); b <= Math.min(gh - 1, gy + span); b++) {
          for (let a = Math.max(0, gx - span); a <= Math.min(gw - 1, gx + span); a++) {
            for (let j = head[b * gw + a]; j >= 0; j = next[j]) {
              if (j === e) continue;
              const dx = x[j] - x[e], dy = y[j] - y[e];
              const d2 = dx * dx + dy * dy;
              if (d2 >= far * far) continue;
              const d = Math.sqrt(d2);
              if (e !== 0 && j !== 0 && !hold && d > 1e-6) {
                const need = 0.42 * (gap[e] + gap[j]);
                if (d < need) { px -= dx / d * (need - d); py -= dy / d * (need - d); }
                // Balls that touch are parted at once.
                const touch = 1.1 * (r[e] + r[j]) + 1;
                if (d < touch) { x[e] -= dx / d * (touch - d) * 0.5; y[e] -= dy / d * (touch - d) * 0.5; }
              }
              // The nearest `most` + 1 so far, in order.
              let at = held <= most ? held++ : most + 1;
              if (at === most + 1 && d >= best[2 * most]) continue;
              if (at === most + 1) at = most;
              while (at > 0 && best[2 * (at - 1)] > d) {
                best[2 * at] = best[2 * (at - 1)];
                best[2 * at + 1] = best[2 * (at - 1) + 1];
                at--;
              }
              best[2 * at] = d;
              best[2 * at + 1] = j;
            }
          }
        }
        if (e !== 0 && !hold) { vx[e] += px * 12 * dt; vy[e] += py * 12 * dt; }

        // The last of the nearest gives way to the one after it only if that
        // one is held already and hardly further off.
        if (held > most) {
          const last = best[2 * (most - 1) + 1], spare = best[2 * most + 1];
          if (this.slotOf(e, spare) >= 0 && this.slotOf(e, last) < 0 && best[2 * most] < best[2 * (most - 1)] * 1.15) {
            best[2 * (most - 1) + 1] = spare;
          }
          held = most;
        }

        // What it chose again it keeps; what it newly chose takes a free
        // slot, or the faintest of those fraying.
        for (let s = base; s < base + S; s++) keep[s] = 0;
        for (let h = 0; h < held; h++) {
          const s = this.slotOf(e, best[2 * h + 1]);
          if (s >= 0) keep[s] = 1;
        }
        for (let h = 0; h < held; h++) {
          const j = best[2 * h + 1];
          if (this.slotOf(e, j) >= 0) continue;
          let slot = -1, faintest = 2;
          for (let s = base; s < base + S; s++) {
            if (mate[s] < 0) { slot = s; break; }
            if (!keep[s] && sig[s] < faintest) { faintest = sig[s]; slot = s; }
          }
          if (slot < 0) continue;
          mate[slot] = j;
          sig[slot] = settle || calm ? 1 : 0;
          keep[slot] = 1;
          this.w1[slot] = this.w2[slot] = 0;
        }
      }

      // Every frame each thread grows towards what was last chosen, or frays,
      // and once gone frees its slot.
      for (let s = base; s < base + S; s++) {
        if (mate[s] < 0) continue;
        sig[s] = keep[s] ? Math.min(1, sig[s] + up) : sig[s] - down;
        if (sig[s] <= 0) { mate[s] = -1; sig[s] = 0; keep[s] = 0; }
      }
    }

    // ---- each free eye steps, and keeps to its ground ----
    // Never into the raster, nor into the clear ring: at either it slides
    // along instead.
    if (!hold) {
      const depth = this.depth, cx = this.cx, cy = this.cy, ring = this.ring;
      for (let e = 1; e <= F; e++) {
        let px = x[e] + vx[e] * dt, py = y[e] + vy[e] * dt;
        const within = this.face(px, py);
        if (within < depth) {
          const nx = this.inX, ny = this.inY;
          px += nx * (depth - within);
          py += ny * (depth - within);
          const outward = -(vx[e] * nx + vy[e] * ny);
          if (outward > 0) { vx[e] += outward * nx; vy[e] += outward * ny; }
        }
        const dx = px - cx, dy = py - cy, d = Math.sqrt(dx * dx + dy * dy) || 1;
        if (d < ring) {
          px = cx + dx / d * ring;
          py = cy + dy / d * ring;
          const inward = (vx[e] * dx + vy[e] * dy) / d;
          if (inward < 0) { vx[e] -= inward * dx / d; vy[e] -= inward * dy / d; }
        }
        x[e] = px;
        y[e] = py;
      }
    }
    this.size();

    // ---- where each whole eye looks ----
    // At the great eye, or away from it. The bigger the eye, the more often it
    // looks somewhere new, the quicker it gets there and the further away it
    // looks: the small ones gaze, the big ones cannot settle.
    if (calm) return;
    const { gaze, gazeTo, gazeAt } = this;
    for (let e = 1; e <= F; e++) {
      if (r[e] < this.WHOLE) continue;
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

  /** The inks of the eyes and of the page, as the theme has them now. */
  inks() {
    return { ball: Ink.of('eyeBall'), iris: Ink.of('eyeIris'), pupil: Ink.of('eyePupil'),
             rim: Ink.of('eyeRim'), link: Ink.of('eyeLink'), stage: Ink.of('stage') };
  },

  /** A colour written as #rgb or #rrggbb, as red, green and blue. */
  rgb(colour) {
    const hex = String(colour).trim().replace(/^#/, '');
    const full = hex.length === 3 ? [...hex].map(c => c + c).join('') : hex;
    const [r, g, b] = [0, 2, 4].map(i => parseInt(full.slice(i, i + 2), 16));
    return [r, g, b].map(v => (Number.isFinite(v) ? v : 128));
  },

  /**
   * The tube, which never moves, drawn once into a picture kept aside —
   * draw() lays it down every frame, faded in as the web comes up. Its glass
   * is a wash of the eyes' colour, a line along every row of the raster and a
   * still grain, cut down to how strongly the glass shows at each point; on it
   * the raster's pixel-eyes, each an iris and a pupil, with the threads that
   * start among the loosest, all bent with the bulge; and round it the rim,
   * whose shadow darkens the glass along its edge — on Paper, where a dark
   * shadow would be dirt, the rim is pale and its shadow the eyes' blue — with
   * a thin catch of light along the glass's edge and, on a dark screen, a
   * faint sheen high on the left. Beside it, at a quarter size, the glass's
   * colour, which draw() lays over the free eyes near the edge.
   */
  paintStill() {
    if (!this.ctx || !this.n) return;
    const W = this.width, H = this.height, dpr = this.dpr, P = this.pitch;
    const ink = this.inks();
    const [ir, ig, ib] = this.rgb(ink.iris);
    const [sr, sg, sb] = this.rgb(ink.stage);
    const light = sr * 0.299 + sg * 0.587 + sb * 0.114 > 128;
    const make = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };

    const cache = this.cache || (this.cache = make(1, 1));
    cache.width = this.canvas.width;
    cache.height = this.canvas.height;
    const g = cache.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);

    // How strongly the glass shows at each point of it, and whether the
    // raster lies there: worked out on a coarse grid, the bulge undone at
    // each, and stretched — it changes slowly, and the stretching smooths it.
    // The rim's shadow is worked out here too: dark just inside the outline,
    // gone a raster's depth in.
    const Q = 6, mw = Math.ceil(W / Q) + 1, mh = Math.ceil(H / Q) + 1;
    const strength = new Float32Array(mw * mh), raster = new Uint8Array(mw * mh), shadow = new Float32Array(mw * mh);
    for (let v = 0; v < mh; v++) {
      for (let u = 0; u < mw; u++) {
        this.unwarp(u * Q, v * Q);
        const inset = this.face(this.atX, this.atY), i = v * mw + u;
        strength[i] = this.glass(inset);
        raster[i] = inset < this.depth - 0.1 * P ? 1 : 0;
        const fall = Math.max(0, Math.min(1, inset / this.depth));
        shadow[i] = (1 - fall) * (1 - fall) * (1 - fall);
      }
    }
    const mask = make(mw, mh), mc = mask.getContext('2d'), md = mc.createImageData(mw, mh);
    for (let i = 0; i < mw * mh; i++) md.data[4 * i + 3] = Math.round(255 * strength[i]);
    mc.putImageData(md, 0, 0);

    // The glass, on a layer of its own: the wash, a line along every row of
    // the raster bent with the bulge, and the grain, a small tile of specks
    // laid edge to edge; then cut down to its strength.
    const layer = make(cache.width, cache.height), l = layer.getContext('2d');
    l.setTransform(dpr, 0, 0, dpr, 0, 0);
    l.fillStyle = ink.iris;
    l.globalAlpha = light ? 0.09 : 0.07;
    l.fillRect(0, 0, W, H);
    l.strokeStyle = ink.iris;
    l.lineWidth = 1;
    l.globalAlpha = light ? 0.22 : 0.18;
    l.beginPath();
    const STEP = 24;
    for (let row = P / 2; row < H; row += P) {
      for (let col = 0; col < W + STEP; col += STEP) {
        this.warp(Math.min(col, W), row);
        if (col === 0) l.moveTo(this.atX, this.atY);
        else l.lineTo(this.atX, this.atY);
      }
    }
    l.stroke();
    const TILE = 192;
    const grain = make(TILE, TILE), gc = grain.getContext('2d'), gd = gc.createImageData(TILE, TILE);
    const rand = seededRandom(977);
    for (let i = 0; i < TILE * TILE; i++) {
      const speck = rand();
      gd.data[4 * i] = ir;
      gd.data[4 * i + 1] = ig;
      gd.data[4 * i + 2] = ib;
      gd.data[4 * i + 3] = Math.round(255 * speck * speck * speck * (light ? 0.45 : 0.5));
    }
    gc.putImageData(gd, 0, 0);
    // Each speck two pixels across, as a tube's grain is coarser than its glass.
    const specks = l.createPattern(grain, 'repeat');
    specks.setTransform(new DOMMatrix().scale(2));
    l.globalAlpha = 1;
    l.fillStyle = specks;
    l.fillRect(0, 0, W, H);
    l.globalCompositeOperation = 'destination-in';
    l.setTransform(1, 0, 0, 1, 0, 0);
    l.drawImage(mask, 0, 0, mw * Q * dpr, mh * Q * dpr);
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.drawImage(layer, 0, 0);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);

    const { sx, sy, r, alpha, F, n, fixed } = this;
    const LEVELS = 8;
    const shade = (items, paint, done) => {
      const buckets = Array.from({ length: LEVELS }, () => []);
      for (const item of items) buckets[Math.min(LEVELS - 1, Math.floor(item.a * LEVELS))].push(item);
      buckets.forEach((list, b) => {
        if (!list.length) return;
        g.globalAlpha = (b + 0.5) / LEVELS;
        g.beginPath();
        for (const item of list) paint(item);
        done();
      });
    };

    // The threads that start among the loosest pixels, straight and faint.
    g.strokeStyle = ink.link;
    g.lineWidth = 0.8;
    const lines = Array.from({ length: LEVELS }, () => []);
    for (let f = 0; f < fixed.length; f += 2) {
      const a = fixed[f], b = fixed[f + 1];
      const strong = Math.min(alpha[a], alpha[b]) * 0.5 * this.grown(Math.min(r[a], r[b]));
      lines[Math.min(LEVELS - 1, Math.floor(strong * LEVELS))].push(a, b);
    }
    lines.forEach((list, b) => {
      if (!list.length) return;
      g.globalAlpha = (b + 0.5) / LEVELS;
      g.beginPath();
      for (let i = 0; i < list.length; i += 2) { g.moveTo(sx[list[i]], sy[list[i]]); g.lineTo(sx[list[i + 1]], sy[list[i + 1]]); }
      g.stroke();
    });

    // The pixels: each a tiny eye, an iris and a pupil, in its row.
    const pixels = [];
    for (let e = 1 + F; e < n; e++) pixels.push({ e, a: alpha[e] });
    g.fillStyle = ink.iris;
    shade(pixels, ({ e }) => { g.moveTo(sx[e] + r[e], sy[e]); g.arc(sx[e], sy[e], r[e], 0, Math.PI * 2); }, () => g.fill());
    g.fillStyle = ink.pupil;
    shade(pixels, ({ e }) => { g.moveTo(sx[e] + r[e] * 0.45, sy[e]); g.arc(sx[e], sy[e], r[e] * 0.45, 0, Math.PI * 2); },
          () => g.fill());

    // The face's outline on the glass: the rounded square, bent with the bulge.
    const outline = [];
    for (let i = 0; i < 400; i++) {
      const a = (i / 400) * Math.PI * 2, c = Math.cos(a), s = Math.sin(a);
      this.warp(W / 2 + Math.cbrt(c) * this.ax, H / 2 + Math.cbrt(s) * this.ay);
      outline.push(this.atX, this.atY);
    }
    const trace = c => {
      c.moveTo(outline[0], outline[1]);
      for (let i = 2; i < outline.length; i += 2) c.lineTo(outline[i], outline[i + 1]);
      c.closePath();
    };

    // The rim's shadow on the glass: black, or on Paper the eyes' blue.
    const dark = make(mw, mh), hc = dark.getContext('2d'), hd = hc.createImageData(mw, mh);
    const [hr, hg, hb] = light ? [ir, ig, ib] : [0, 0, 0];
    for (let i = 0; i < mw * mh; i++) {
      hd.data[4 * i] = hr;
      hd.data[4 * i + 1] = hg;
      hd.data[4 * i + 2] = hb;
      hd.data[4 * i + 3] = Math.round(255 * shadow[i] * (light ? 0.55 : 0.85));
    }
    hc.putImageData(hd, 0, 0);
    g.globalAlpha = 1;
    g.drawImage(dark, 0, 0, mw * Q, mh * Q);

    // The rim: everything beyond the glass — on a dark screen the page's own
    // dark, deeper still, as a tube's black; on Paper a pale blue-grey.
    const [rr, rg, rb] = light
      ? [sr * 0.82 + ir * 0.18, sg * 0.82 + ig * 0.18, sb * 0.82 + ib * 0.18]
      : [sr * 0.3, sg * 0.3, sb * 0.3];
    g.fillStyle = `rgb(${Math.round(rr)}, ${Math.round(rg)}, ${Math.round(rb)})`;
    g.beginPath();
    g.rect(-40, -40, W + 80, H + 80);
    trace(g);
    g.fill('evenodd');
    // A thin catch of light along the glass's edge, in the eyes' white, or
    // on Paper their blue.
    g.strokeStyle = light ? ink.iris : ink.ball;
    g.globalAlpha = light ? 0.4 : 0.14;
    g.lineWidth = 1.2;
    g.beginPath();
    trace(g);
    g.stroke();
    // A faint sheen on a dark glass, high on its left.
    if (!light) {
      g.save();
      g.beginPath();
      trace(g);
      g.clip();
      const sheen = g.createRadialGradient(W * 0.2, H * 0.12, 0, W * 0.2, H * 0.12, W * 0.45);
      sheen.addColorStop(0, Ink.alpha(ink.ball, 0.05));
      sheen.addColorStop(1, Ink.alpha(ink.ball, 0));
      g.globalAlpha = 1;
      g.fillStyle = sheen;
      g.fillRect(0, 0, W, H);
      g.restore();
    }
    g.globalAlpha = 1;

    // The glass's own colour, laid over the free eyes and their threads near
    // the edge (draw()): an eye just come off the raster still shows in the
    // screen's colour, and turns white only further in. From the inside of
    // the raster on, so the pixels keep their dark pupils.
    const tint = this.tint || (this.tint = make(1, 1));
    tint.width = mw;
    tint.height = mh;
    const tc = tint.getContext('2d'), td = tc.createImageData(mw, mh);
    for (let i = 0; i < mw * mh; i++) {
      if (raster[i]) continue;
      td.data[4 * i] = ir;
      td.data[4 * i + 1] = ig;
      td.data[4 * i + 2] = ib;
      td.data[4 * i + 3] = Math.round(255 * strength[i] * (light ? 0.6 : 0.7));
    }
    tc.putImageData(td, 0, 0);
    this.glassScale = Q;
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
    const { F, x, y, sx, sy, r, alpha, mate, sig, w1, w2, from, shown } = this;
    const W = this.width, H = this.height, dpr = this.dpr, S = this.SLOTS;
    const LEVELS = 8, TAU = Math.PI * 2;

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    // ---- the tube, still ----
    if (this.cache) {
      ctx.globalAlpha = fade;
      ctx.drawImage(this.cache, 0, 0, W, H);
    }

    // Where each free eye shows on the bulging glass.
    for (let e = 1; e <= F; e++) {
      this.warp(x[e], y[e]);
      sx[e] = this.atX;
      sy[e] = this.atY;
    }

    // ---- the threads, in a few strengths so each is one stroke call ----
    // A thread held from both ends is drawn once, from the lower, as grown as
    // its more grown end. One coming or going is part grown: it reaches out of
    // the eye that chose it towards the other, brightening as it goes, and
    // draws back into that eye as it frays.
    const [threads, dots, small, whole] = this.bins;
    for (const bin of this.bins) for (const list of bin) list.length = 0;
    this.large.length = 0;
    for (let e = 0; e <= F; e++) {
      for (let s = e * S; s < e * S + S; s++) {
        const p = mate[s];
        if (p < 0) continue;
        let grownTo = sig[s], back = false;
        if (p <= F) {
          const o = this.slotOf(p, e);
          if (o >= 0) {
            if (p < e) continue;
            if (sig[o] > grownTo) { grownTo = sig[o]; back = true; }
          }
        }
        if (grownTo <= 0) continue;
        const strength = Math.min(alpha[e], alpha[p]) * 0.5 * this.grown(Math.min(r[e], r[p]))
                       * Math.min(1, grownTo * 2) * fade;
        if (strength < 0.016) continue;
        // The tremble: two points along the thread, each pushed sideways by a
        // little more or less every frame and drawn back towards the line. A
        // short thread would tremble by less than a pixel, so it is drawn straight.
        const lx = sx[p] - sx[e], ly = sy[p] - sy[e], length = Math.sqrt(lx * lx + ly * ly);
        if (!calm && length > 12) {
          const amp = Math.min(3, length * 0.04);
          w1[s] = Math.max(-amp, Math.min(amp, w1[s] * 0.9 + (Math.random() - 0.5) * amp * 0.5));
          w2[s] = Math.max(-amp, Math.min(amp, w2[s] * 0.9 + (Math.random() - 0.5) * amp * 0.5));
        }
        from[s] = back ? 1 : 0;
        shown[s] = grownTo;
        threads[Math.min(LEVELS - 1, Math.floor(strength / 0.5 * LEVELS))].push(s);
      }
    }
    ctx.strokeStyle = ink.link;
    ctx.lineWidth = 1;
    threads.forEach((list, level) => {
      if (!list.length) return;
      ctx.globalAlpha = ((level + 0.5) / LEVELS) * 0.5;
      ctx.beginPath();
      for (const s of list) {
        const owner = Math.floor(s / S), mateOf = mate[s];
        const a = from[s] ? mateOf : owner, b = from[s] ? owner : mateOf;
        const g = shown[s], grow = g >= 1 ? 1 : g * g * (3 - 2 * g);
        const px = sx[a], py = sy[a], dx = (sx[b] - px) * grow, dy = (sy[b] - py) * grow;
        ctx.moveTo(px, py);
        const length = Math.sqrt((sx[b] - px) ** 2 + (sy[b] - py) ** 2);
        if (length > 12) {
          const nx = -dy / length, ny = dx / length;
          ctx.lineTo(px + dx / 3 + nx * w1[s], py + dy / 3 + ny * w1[s]);
          ctx.lineTo(px + dx * 2 / 3 + nx * w2[s], py + dy * 2 / 3 + ny * w2[s]);
        }
        ctx.lineTo(px + dx, py + dy);
      }
      ctx.stroke();
    });

    // ---- the free eyes, sorted by size and strength ----
    const { ix, iy, open, gaze, blink, blinkAt } = this;
    // Gazes are aimed where the great eye shows.
    const cx = sx[0], cy = sy[0];
    for (let e = 1; e <= F; e++) {
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
      const look = Math.atan2(cy - sy[e], cx - sx[e]) + (calm ? 0 : gaze[e])
                 + (shake ? (Math.random() - 0.5) * shake * 0.3 : 0);
      const off = (0.3 + 0.06 * fear) * re;
      ix[e] = sx[e] + Math.cos(look) * off + (shake ? (Math.random() - 0.5) * shake * re * 0.05 : 0);
      iy[e] = sy[e] + Math.sin(look) * off * open[e] + (shake ? (Math.random() - 0.5) * shake * re * 0.05 : 0);
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
        ctx.rect(Math.round(sx[e] - d / 2), Math.round(sy[e] - d / 2), d, d);
      }
      ctx.fill();
    });

    // The small eyes, just off the raster, still look like its pixels: an
    // iris and a pupil, nothing more at this size.
    small.forEach((list, level) => {
      if (!list.length) return;
      ctx.globalAlpha = (level + 0.5) / LEVELS;
      ctx.fillStyle = ink.iris;
      ctx.beginPath();
      for (const e of list) { ctx.moveTo(sx[e] + r[e], sy[e]); ctx.arc(sx[e], sy[e], r[e], 0, TAU); }
      ctx.fill();
      // A pupil a pixel or two across is as round as a square of that size.
      ctx.fillStyle = ink.pupil;
      ctx.beginPath();
      for (const e of list) { const p = r[e] * 0.8; ctx.rect(sx[e] - p / 2, sy[e] - p / 2, p, p); }
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
      for (const e of list) oval(sx[e], sy[e], r[e], open[e]);
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
        oval(sx[e] - r[e] * 0.26, sy[e] - r[e] * 0.28 * open[e], r[e] * 0.09, 1);
      }
      ctx.fill();
    });

    // The big ones, each on its own.
    for (const e of this.large) {
      this.eye(ctx, sx[e], sy[e], r[e], open[e], alpha[e] * fade, ink, { jx: ix[e] - sx[e], jy: iy[e] - sy[e] });
    }

    // ---- the glass's colour, over all that lies near the edge ----
    if (this.tint) {
      ctx.globalCompositeOperation = 'source-atop';
      ctx.globalAlpha = 1;
      ctx.drawImage(this.tint, 0, 0, this.tint.width * this.glassScale, this.tint.height * this.glassScale);
      ctx.globalCompositeOperation = 'source-over';
    }

    // ---- the great eye, looking at you ----
    const m = this.big;
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
    const glow = ctx.createRadialGradient(cx, cy, m * 0.8, cx, cy, m * 2.4);
    glow.addColorStop(0, ink.iris);
    glow.addColorStop(1, Ink.alpha(ink.iris, 0));
    ctx.fillStyle = glow;
    ctx.beginPath(); ctx.arc(cx, cy, m * 2.4, 0, TAU); ctx.fill();

    this.eye(ctx, cx, cy, m, 1, fade, ink, { jx, jy, breathe, spokes: true });
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
