/*
 * The front page's screen: an old tube television, laid over whichever
 * backdrop is showing and under the words, so the world and the eyes alike are
 * seen through its glass. The words stay crisp above it.
 *
 * The glass is a rounded square — |X|⁶ + |Y|⁶ = 1 on the stage's half-sizes —
 * that bulges a little, so what is drawn near its edge bends in towards the
 * corners (eyes.js draws its picture through warp()). On it lie still scan
 * lines, bent with the bulge, a fine grain of phosphor and a glow of the eyes'
 * colour, as a lit tube has — faint in the middle, round the great eye and
 * the words, and growing steadily stronger out to the glass's edge, where on
 * Paper it deepens into blue — and, on a dark screen, a faint sheen high on
 * the left. There is no frame: the same look runs on past the glass into the
 * window's corners, at its full strength, so no corner is left empty, and only
 * the bending rows and the rounded line where the eyes come loose tell where
 * the tube is. The faintest tone darkens (on Paper) or lightens (on a dark
 * screen) the very edge of the window, all round alike.
 *
 * Nothing on it moves. It is drawn once for each size and theme into a canvas
 * of its own, and costs nothing a frame.
 */
const Tube = {
  /** How far the picture bends in towards the corners of the bulging glass, as a share of the half-width. */
  BULGE: 0.04,

  /** How strongly the tube shows in the very middle of the glass, against its full strength at the edge. */
  FAINT: 0.15,

  canvas: null,
  host: null,
  width: 0,
  height: 0,
  ax: 0,
  ay: 0,
  pitch: 6,

  init(canvas, host) {
    this.canvas = canvas;
    this.host = host;
  },

  /**
   * The glass for a screen of this size, and the pitch both its scan lines
   * and the eyes' raster keep to: a little coarser on a bigger screen.
   */
  shape(width, height) {
    this.width = width;
    this.height = height;
    this.ax = width / 2;
    this.ay = height / 2;
    this.pitch = Math.max(4, Math.min(7, Math.round(Math.min(width, height) / 170)));
  },

  /**
   * How far a point of the picture lies inside the glass, near enough: how far
   * short of the outline it is, over how fast that changes. Negative outside
   * it. Leaves the way inwards in `inX`, `inY`.
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

  /** How far out a point of the picture lies on the glass: 0 in the middle, 1 on the outline. */
  round(x, y) {
    const X = (x - this.width / 2) / this.ax, Y = (y - this.height / 2) / this.ay;
    const X2 = X * X, Y2 = Y * Y;
    return Math.cbrt(Math.sqrt(X2 * X2 * X2 + Y2 * Y2 * Y2));
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

  /** A colour written as #rgb or #rrggbb, as red, green and blue. */
  rgb(colour) {
    const hex = String(colour).trim().replace(/^#/, '');
    const full = hex.length === 3 ? [...hex].map(c => c + c).join('') : hex;
    const [r, g, b] = [0, 2, 4].map(i => parseInt(full.slice(i, i + 2), 16));
    return [r, g, b].map(v => (Number.isFinite(v) ? v : 128));
  },

  /** Laid out and drawn for the stage as it now is: on arrival, and whenever it changes size. */
  resize() {
    const canvas = this.canvas, host = this.host;
    if (!canvas || !host) return;
    const width = host.clientWidth, height = host.clientHeight;
    if (!(width > 0 && height > 0)) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (this.painted === `${width}x${height}x${dpr}`) return;
    this.shape(width, height);
    this.dpr = dpr;
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    this.paint();
  },

  /** The theme changed: the same glass in its new inks. */
  repaint() {
    if (this.painted) this.paint();
  },

  paint() {
    const canvas = this.canvas;
    if (!canvas || !this.width) return;
    const g = canvas.getContext('2d'), W = this.width, H = this.height, dpr = this.dpr;
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, H);
    const iris = Ink.of('eyeIris'), stage = Ink.of('stage'), ball = Ink.of('eyeBall');
    const [sr, sg, sb] = this.rgb(stage);
    const light = sr * 0.299 + sg * 0.587 + sb * 0.114 > 128;

    // ---- the glass, and on past it into the corners ----
    // How strongly the tube shows at each point: faint in the middle, round
    // the great eye and the words, growing steadily stronger out to the
    // glass's rounded edge, where the eyes' raster is, and at its full
    // strength from there to the window's edge. Worked out on a coarse grid,
    // the bulge undone at each point, and stretched: it changes slowly.
    const make = (w, h) => { const c = document.createElement('canvas'); c.width = w; c.height = h; return c; };
    const Q = 6, mw = Math.ceil(W / Q) + 1, mh = Math.ceil(H / Q) + 1;
    const mask = make(mw, mh), mc = mask.getContext('2d'), md = mc.createImageData(mw, mh);
    for (let v = 0; v < mh; v++) {
      for (let u = 0; u < mw; u++) {
        this.unwarp(u * Q, v * Q);
        const out = Math.min(1, this.round(this.atX, this.atY));
        md.data[4 * (v * mw + u) + 3] = Math.round(255 * (this.FAINT + (1 - this.FAINT) * out * out * out));
      }
    }
    mc.putImageData(md, 0, 0);

    // The tube's look at full strength, on a layer of its own: a wash of the
    // eyes' colour, the glow of the lit glass, which on Paper deepens it into
    // blue; a fine still grain, the phosphor; and the scan lines, one at every
    // half step of the eyes' raster, so they run through and between its rows,
    // bent with the bulge — dark on a dark screen, the eyes' blue on Paper.
    // Then cut down to the strength at each point.
    const layer = make(canvas.width, canvas.height), l = layer.getContext('2d');
    l.setTransform(dpr, 0, 0, dpr, 0, 0);
    l.fillStyle = iris;
    l.globalAlpha = light ? 0.3 : 0.16;
    l.fillRect(0, 0, W, H);
    const TILE = 160, [ir, ig, ib] = this.rgb(iris);
    const grain = make(TILE, TILE), gc = grain.getContext('2d'), gd = gc.createImageData(TILE, TILE);
    const rand = seededRandom(613);
    for (let i = 0; i < TILE * TILE; i++) {
      const speck = rand();
      gd.data[4 * i] = ir;
      gd.data[4 * i + 1] = ig;
      gd.data[4 * i + 2] = ib;
      gd.data[4 * i + 3] = Math.round(255 * speck * speck * speck * 0.5);
    }
    gc.putImageData(gd, 0, 0);
    const specks = l.createPattern(grain, 'repeat');
    specks.setTransform(new DOMMatrix().scale(2));
    l.globalAlpha = 1;
    l.fillStyle = specks;
    l.fillRect(0, 0, W, H);
    const STEP = 24;
    l.strokeStyle = light ? iris : stage;
    l.globalAlpha = light ? 0.32 : 0.5;
    l.lineWidth = 1;
    l.beginPath();
    const overX = this.BULGE * W / 2 + this.pitch, overY = this.BULGE * H / 2 + this.pitch;
    for (let row = -Math.ceil(overY / (this.pitch / 2)) * (this.pitch / 2); row <= H + overY; row += this.pitch / 2) {
      for (let col = -overX; col < W + overX + STEP; col += STEP) {
        this.warp(Math.min(col, W + overX), row);
        if (col === -overX) l.moveTo(this.atX, this.atY);
        else l.lineTo(this.atX, this.atY);
      }
    }
    l.stroke();
    l.globalCompositeOperation = 'destination-in';
    l.globalAlpha = 1;
    l.setTransform(1, 0, 0, 1, 0, 0);
    l.drawImage(mask, 0, 0, mw * Q * dpr, mh * Q * dpr);

    g.save();
    g.setTransform(1, 0, 0, 1, 0, 0);
    g.drawImage(layer, 0, 0);
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    // A faint sheen high on the left of a dark glass, where a window would
    // catch it.
    if (!light) {
      const sheen = g.createRadialGradient(W * 0.2, H * 0.12, 0, W * 0.2, H * 0.12, W * 0.45);
      sheen.addColorStop(0, Ink.alpha(ball, 0.05));
      sheen.addColorStop(1, Ink.alpha(ball, 0));
      g.fillStyle = sheen;
      g.fillRect(0, 0, W, H);
    }
    g.restore();

    // ---- toward the window's edge ----
    // The faintest tone along every edge of the window alike — a shade darker
    // on a white page, a shade lighter on a dark one — taken where two edges
    // meet as the stronger of the two rather than both, so a corner reads as
    // any other stretch of edge.
    g.save();
    const [er, eg, eb] = light ? [sr * 0.9, sg * 0.9, sb * 0.9]
                               : [sr + (255 - sr) * 0.08, sg + (255 - sg) * 0.08, sb + (255 - sb) * 0.08];
    const edge = a => `rgba(${Math.round(er)}, ${Math.round(eg)}, ${Math.round(eb)}, ${a})`;
    const band = Math.min(W, H) * 0.05;
    const sides = [[0, 0, band, 0], [W, 0, W - band, 0], [0, 0, 0, band], [0, H, 0, H - band]];
    g.globalCompositeOperation = light ? 'darken' : 'lighten';
    for (const [x0, y0, x1, y1] of sides) {
      const fall = g.createLinearGradient(x0, y0, x1, y1);
      fall.addColorStop(0, edge(0.5));
      fall.addColorStop(1, edge(0));
      g.fillStyle = fall;
      g.fillRect(0, 0, W, H);
    }
    g.restore();
    this.painted = `${W}x${H}x${dpr}`;
  }
};
