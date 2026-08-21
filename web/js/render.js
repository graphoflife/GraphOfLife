/*
 * Canvas renderer for a single frame, in two or three dimensions.
 *
 * Everything visual is driven by a settings object: which quantity drives node
 * colour, which drives node size, the same for edges, background style, and
 * opacities. The renderer only maps numbers to pixels — it never decides what
 * a value means.
 *
 * 3D is a projection, not a separate engine. Points are rotated by yaw and
 * pitch, given a perspective divide, and drawn back-to-front so nearer nodes
 * cover farther ones. In 2D the same path runs with the rotation skipped, so
 * there is one drawing routine rather than two.
 */
class GraphRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    // Where the camera is, and where it is heading. Fitting only ever moves
    // the target; the view chases it under stepCamera, so the framing glides
    // rather than jumping every time the graph shifts.
    this.view = { scale: 1, offsetX: 0, offsetY: 0 };
    this.targetView = { scale: 1, offsetX: 0, offsetY: 0 };
    this.cameraVelocity = { logScale: 0, x: 0, y: 0 };
    // Critically damped: the camera reaches its target in about 27 frames
    // without sailing past it. Overshoot would swing the graph out of frame and
    // back, which reads as a wobble rather than as smoothness.
    this.cameraStiffness = 0.17;
    this.cameraDamping = 0.5;

    this.dpr = window.devicePixelRatio || 1;

    // A ceiling on the pixel ratio. The Viewer wants every pixel the screen
    // has; the front page's backdrop does not, and on a retina display the
    // difference is four times the fill for something sitting behind a title.
    this.maxDpr = Infinity;

    this.mode3D = false;
    this.yaw = 0.6;
    this.pitch = -0.35;
    // Distance of the eye from the scene centre, in world units. Larger means
    // a flatter, more orthographic look.
    this.cameraDistance = 900;

  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = Math.min(this.maxDpr, window.devicePixelRatio || 1);
    this.canvas.width = Math.max(1, Math.floor(rect.width * this.dpr));
    this.canvas.height = Math.max(1, Math.floor(rect.height * this.dpr));
    this.cssWidth = rect.width;
    this.cssHeight = rect.height;
  }

  setMode3D(on) {
    this.mode3D = Boolean(on);
  }

  rotate(dYaw, dPitch) {
    this.yaw += dYaw;
    // Stop short of straight up or down; passing the pole flips the scene.
    const limit = Math.PI / 2 - 0.05;
    this.pitch = Math.max(-limit, Math.min(limit, this.pitch + dPitch));
  }

  /**
   * Frame the graph as tightly as the canvas allows.
   *
   * Measured from where the nodes actually land on screen rather than from
   * their world-space bounding box. In 3D those differ: rotation swings the
   * depth axis into view and the perspective divide shrinks far nodes, so a
   * world-space box has to be padded generously to avoid clipping and ends up
   * leaving a wide empty margin. Projecting first removes the guesswork.
   *
   * Because the perspective divide depends on the scale being solved for, 3D
   * takes a second pass to settle; in 2D the projection is linear and one is
   * exact.
   */
  fitToContent(layout, padding = 0.015, snap = false) {
    if (!layout || !layout.positions || !layout.positions.length) return;
    const target = this.computeFitTarget(layout, padding);
    if (!target) return;

    this.targetView = target;
    if (snap) this.snapToTarget();
  }

  /**
   * Work out the tight framing without moving the camera there.
   *
   * The solver has to project, and projecting reads the current view, so it
   * borrows the view, converges, then hands it back untouched.
   */
  computeFitTarget(layout, padding = 0.015) {
    if (!layout || !layout.ids.length || !this.cssWidth) return null;
    if (!layout.positions || layout.positions.length < 3) return null;

    const saved = { ...this.view };
    const restore = () => { this.view = saved; };
    const passes = this.mode3D ? 2 : 1;

    for (let pass = 0; pass < passes; pass++) {
      const box = this._projectedBounds(layout);
      if (!box) { restore(); return null; }

      const availableW = this.cssWidth * (1 - padding * 2);
      const availableH = this.cssHeight * (1 - padding * 2);
      const factor = Math.min(availableW / box.width, availableH / box.height);
      if (!Number.isFinite(factor) || factor <= 0) { restore(); return null; }

      this.view.scale = GraphRenderer.clampScale(this.view.scale * factor);

      // Scaling moves everything, so recentre against the new projection.
      const after = this._projectedBounds(layout);
      if (!after) { restore(); return null; }
      this.view.offsetX += this.cssWidth / 2 - after.centerX;
      this.view.offsetY += this.cssHeight / 2 - after.centerY;
    }

    const target = { ...this.view };
    restore();
    return target;
  }

  static clampScale(s) {
    return Math.max(0.02, Math.min(60, s));
  }

  /** Put the camera on its target immediately, for a first framing. */
  snapToTarget() {
    this.view = { ...this.targetView };
    this.cameraVelocity = { logScale: 0, x: 0, y: 0 };
  }

  /** Stop chasing: whatever the camera is looking at becomes the target. */
  holdCurrentView() {
    this.targetView = { ...this.view };
    this.cameraVelocity = { logScale: 0, x: 0, y: 0 };
  }

  /**
   * Move the camera one step toward its target.
   *
   * A spring with its own velocity rather than a straight interpolation, so
   * the framing eases in and out and keeps gliding for a moment after the
   * target stops — which is what makes it read as smooth even while the graph
   * underneath is still jostling.
   *
   * Scale travels in log space: halving and doubling then take the same time,
   * where a linear zoom would crawl at one end and lurch at the other.
   */
  stepCamera() {
    const target = this.targetView;
    const v = this.cameraVelocity;
    const k = this.cameraStiffness;
    const d = this.cameraDamping;

    const logNow = Math.log(this.view.scale);
    const logTarget = Math.log(target.scale);

    const dLog = logTarget - logNow;
    const dx = target.offsetX - this.view.offsetX;
    const dy = target.offsetY - this.view.offsetY;

    // Once the remaining move is under a pixel, land exactly rather than
    // creeping forever.
    if (Math.abs(dLog) < 1e-4 && Math.abs(dx) < 0.05 && Math.abs(dy) < 0.05) {
      this.snapToTarget();
      return;
    }

    v.logScale = (v.logScale + dLog * k) * d;
    v.x = (v.x + dx * k) * d;
    v.y = (v.y + dy * k) * d;

    this.view.scale = GraphRenderer.clampScale(Math.exp(logNow + v.logScale));
    this.view.offsetX += v.x;
    this.view.offsetY += v.y;
  }

  /** Screen-space extent of every node under the current camera. */
  _projectedBounds(layout) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    const pos = layout.positions;
    const n = Math.min(layout.ids.length, Math.floor(pos.length / 3));
    const point = this._boundsScratch || (this._boundsScratch = { x: 0, y: 0, z: 0 });

    for (let i = 0; i < n; i++) {
      const o = i * 3;
      point.x = pos[o]; point.y = pos[o + 1]; point.z = pos[o + 2];
      const s = this.project(point);
      if (!Number.isFinite(s.x) || !Number.isFinite(s.y)) continue;
      if (s.x < minX) minX = s.x;
      if (s.y < minY) minY = s.y;
      if (s.x > maxX) maxX = s.x;
      if (s.y > maxY) maxY = s.y;
    }
    if (!Number.isFinite(minX)) return null;

    // A single node, or a perfectly flat line, has no extent in one axis.
    const width = Math.max(1e-6, maxX - minX);
    const height = Math.max(1e-6, maxY - minY);
    return { width, height, centerX: (minX + maxX) / 2, centerY: (minY + maxY) / 2 };
  }

  /** Fit the given world-space bounds into the canvas. */
  fit(bounds) {
    const w = bounds.maxX - bounds.minX;
    const h = bounds.maxY - bounds.minY;
    if (w <= 0 || h <= 0) return;

    const scale = Math.min(this.cssWidth / w, this.cssHeight / h);
    this.view.scale = scale;
    this.view.offsetX = this.cssWidth / 2 - (bounds.minX + w / 2) * scale;
    this.view.offsetY = this.cssHeight / 2 - (bounds.minY + h / 2) * scale;
    this.holdCurrentView();
  }

  /**
   * World point to screen point.
   *
   * Returns a depth alongside the coordinates: larger means nearer the eye.
   * In 2D the depth is constant and the perspective divide is skipped.
   */
  project(p) {
    if (!this.mode3D) {
      return {
        x: p.x * this.view.scale + this.view.offsetX,
        y: p.y * this.view.scale + this.view.offsetY,
        depth: 0,
        k: 1
      };
    }

    const cosYaw = Math.cos(this.yaw), sinYaw = Math.sin(this.yaw);
    const cosPitch = Math.cos(this.pitch), sinPitch = Math.sin(this.pitch);

    // Yaw about the vertical axis, then pitch about the horizontal one.
    const x1 = p.x * cosYaw + p.z * sinYaw;
    const z1 = -p.x * sinYaw + p.z * cosYaw;
    const y2 = p.y * cosPitch - z1 * sinPitch;
    const z2 = p.y * sinPitch + z1 * cosPitch;

    // Perspective divide, clamped so a node level with the eye cannot explode.
    const denominator = Math.max(0.25, 1 + (z2 * this.view.scale) / this.cameraDistance);
    const k = 1 / denominator;

    return {
      x: x1 * this.view.scale * k + this.view.offsetX,
      y: y2 * this.view.scale * k + this.view.offsetY,
      depth: -z2,
      k
    };
  }

  toScreen(p) { return this.project(p); }

  toWorld(sx, sy) {
    return {
      x: (sx - this.view.offsetX) / this.view.scale,
      y: (sy - this.view.offsetY) / this.view.scale
    };
  }

  zoomAt(sx, sy, factor) {
    const before = this.toWorld(sx, sy);
    this.view.scale = GraphRenderer.clampScale(this.view.scale * factor);
    const after = this.toWorld(sx, sy);
    this.view.offsetX += (after.x - before.x) * this.view.scale;
    this.view.offsetY += (after.y - before.y) * this.view.scale;
    // Hand control over, or the spring would drag the view back.
    this.holdCurrentView();
  }

  pan(dx, dy) {
    this.view.offsetX += dx;
    this.view.offsetY += dy;
    this.holdCurrentView();
  }

  // ------------------------------------------------------------------
  // Drawing
  // ------------------------------------------------------------------

  draw(frame, metrics, layout, settings) {
    const ctx = this.ctx;
    ctx.save();
    ctx.scale(this.dpr, this.dpr);

    this._background(ctx, settings);

    if (frame) {
      // One projection pass per draw, reused by edges, nodes and picking.
      this._projectFrame(frame, layout);

      if (settings.edgeShow) this._edges(ctx, frame, metrics, settings);
      this._nodes(ctx, frame, metrics, settings);
      // _legend decides which keys it has to draw; the two toggles are
      // independent, so gating the call on one of them would hide the other.
      this._legend(ctx, metrics, settings);
    }

    ctx.restore();
  }

  /**
   * Project every node into parallel arrays.
   *
   * Previously this filled a Map keyed by node id, which the depth sort then
   * read twice per comparison. At eighteen thousand nodes that is half a
   * million hashed lookups inside a comparator, and it dominated the draw.
   * Arrays indexed by the node's position in the frame cost one lookup each,
   * once.
   */
  _projectFrame(frame, layout) {
    const n = frame.ids.length;
    if (!this._sx || this._sx.length < n) {
      this._sx = new Float64Array(n);
      this._sy = new Float64Array(n);
      this._sk = new Float64Array(n);
      this._sDepth = new Float64Array(n);
      this._sOk = new Uint8Array(n);
      this._order = new Int32Array(n);
    }

    // Positions arrive as three floats per node, in the same order as
    // frame.ids, whether the layout ran here or in a worker.
    const pos = layout.positions;
    const have = Math.min(n, Math.floor(pos.length / 3));
    const point = this._scratch || (this._scratch = { x: 0, y: 0, z: 0 });

    for (let i = 0; i < n; i++) {
      if (i >= have) { this._sOk[i] = 0; continue; }
      const o = i * 3;
      point.x = pos[o]; point.y = pos[o + 1]; point.z = pos[o + 2];

      const s = this.project(point);
      this._sx[i] = s.x; this._sy[i] = s.y;
      this._sk[i] = s.k; this._sDepth[i] = s.depth;
      this._sOk[i] = 1;
    }
    this._sCount = n;

    // Edge endpoints as indices, so drawing them needs no id lookups either.
    if (this._edgeFrame !== frame) {
      const index = new Map();
      for (let i = 0; i < n; i++) index.set(frame.ids[i], i);
      const pairs = new Int32Array(frame.edges.length * 2);
      for (let e = 0; e < frame.edges.length; e++) {
        const [a, b] = frame.edges[e];
        const ia = index.get(a), ib = index.get(b);
        pairs[e * 2] = ia === undefined ? -1 : ia;
        pairs[e * 2 + 1] = ib === undefined ? -1 : ib;
      }
      this._edgePairs = pairs;
      this._edgeFrame = frame;
    }
  }

  _background(ctx, s) {
    const w = this.cssWidth, h = this.cssHeight;
    if (s.bgStyle === 'solid') {
      ctx.fillStyle = s.bgColorA;
    } else if (s.bgStyle === 'radial') {
      const g = ctx.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, Math.max(w, h) * 0.72);
      g.addColorStop(0, s.bgColorB);
      g.addColorStop(1, s.bgColorA);
      ctx.fillStyle = g;
    } else {
      const g = ctx.createLinearGradient(0, 0, 0, h);
      g.addColorStop(0, s.bgColorB);
      g.addColorStop(1, s.bgColorA);
      ctx.fillStyle = g;
    }
    ctx.fillRect(0, 0, w, h);
  }

  /**
   * Draw the edges, grouped by colour and width.
   *
   * Stroking each edge on its own costs a `strokeStyle` assignment — which
   * re-parses the colour string — plus a path and a stroke, per edge. At
   * thirty-four thousand edges that measured 96ms a frame against 1.5ms for
   * the flat case, and it was by a wide margin the most expensive thing the
   * renderer did.
   *
   * So edges are quantised into a few dozen colours and a dozen widths and
   * drawn one group at a time, exactly as the nodes already were. Roughly six
   * hundred groups replace thirty-four thousand strokes. The quantisation is
   * invisible: twelve widths spread over a range about a pixel wide, and
   * forty-eight colours out of a map the eye reads as continuous anyway.
   *
   * The values themselves come from the metrics' cached per-edge arrays rather
   * than from a lookup per edge, so the endpoint arithmetic happens once per
   * frame instead of once per draw.
   */
  _edges(ctx, frame, metrics, s) {
    const edges = frame.edges;
    const pairs = this._edgePairs;
    const sx = this._sx, sy = this._sy, sk = this._sk, ok = this._sOk;
    const flat = s.edgeColorBy === 'constant';
    const uniformWidth = s.edgeWidthBy === 'constant';
    const m = edges.length;

    ctx.globalAlpha = s.edgeAlpha;

    // Nothing to group when every edge looks the same: one path, one stroke.
    if (flat && uniformWidth) {
      ctx.strokeStyle = s.edgeFlatColor;
      ctx.lineWidth = s.edgeWidthMin;
      ctx.beginPath();
      for (let e = 0; e < m; e++) {
        const ia = pairs[e * 2], ib = pairs[e * 2 + 1];
        if (ia < 0 || ib < 0 || !ok[ia] || !ok[ib]) continue;
        ctx.moveTo(sx[ia], sy[ia]);
        ctx.lineTo(sx[ib], sy[ib]);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      return;
    }

    const CB = flat ? 1 : 48;
    // In 3D even a constant width varies, since it scales with depth.
    const WB = (uniformWidth && !this.mode3D) ? 1 : 12;
    const GROUPS = CB * WB;

    if (!this._edgeBucket || this._edgeBucket.length < m) {
      this._edgeBucket = new Uint16Array(m);
      this._edgeGrouped = new Int32Array(m);
      this._edgeWidth = new Float32Array(m);
    }
    const bucket = this._edgeBucket, grouped = this._edgeGrouped, widths = this._edgeWidth;

    // ---- colour per edge, as a bucket index ----
    let colorAt;
    if (flat) {
      colorAt = () => 0;
    } else if (s.edgeColorBy === 'source') {
      // Inheriting the node colour means inheriting its bucket.
      colorAt = (e, ia) => metrics.nodeColorNorm(ia);
    } else {
      const values = metrics.scaledEdgeValues(s.edgeColorBy, s.edgeColorLog);
      const [lo, hi] = metrics.edgeRange(s.edgeColorBy, s.edgeColorLog);
      const span = (hi - lo) || 1;
      colorAt = (e) => (values[e] - lo) / span;
    }

    // ---- width per edge ----
    let widthAt;
    if (uniformWidth) {
      widthAt = () => 0;
    } else {
      const values = metrics.scaledEdgeValues(s.edgeWidthBy, s.edgeWidthLog);
      const [lo, hi] = metrics.edgeRange(s.edgeWidthBy, s.edgeWidthLog);
      const span = (hi - lo) || 1;
      widthAt = (e) => (values[e] - lo) / span;
    }

    const wMin = s.edgeWidthMin, wSpan = s.edgeWidthMax - s.edgeWidthMin;
    const mode3D = this.mode3D;

    let live = 0, loW = Infinity, hiW = -Infinity;
    for (let e = 0; e < m; e++) {
      const ia = pairs[e * 2], ib = pairs[e * 2 + 1];
      if (ia < 0 || ib < 0 || !ok[ia] || !ok[ib]) { bucket[e] = 0xffff; continue; }

      let width = wMin + wSpan * Math.min(1, Math.max(0, widthAt(e)));
      if (mode3D) width *= (sk[ia] + sk[ib]) / 2;
      width = Math.max(0.05, width);
      widths[e] = width;
      if (width < loW) loW = width;
      if (width > hiW) hiW = width;

      const t = Math.min(1, Math.max(0, colorAt(e, ia)));
      bucket[e] = ((t * (CB - 1)) | 0) * WB;   // width part added below
      live++;
    }
    if (!live) { ctx.globalAlpha = 1; return; }

    const wRange = (hiW - loW) || 1;
    if (WB > 1) {
      for (let e = 0; e < m; e++) {
        if (bucket[e] === 0xffff) continue;
        let wb = ((widths[e] - loW) / wRange * (WB - 1)) | 0;
        if (wb < 0) wb = 0; else if (wb > WB - 1) wb = WB - 1;
        bucket[e] += wb;
      }
    }

    // Counting sort into groups, so each is one contiguous run.
    const counts = new Int32Array(GROUPS + 1);
    for (let e = 0; e < m; e++) if (bucket[e] !== 0xffff) counts[bucket[e] + 1]++;
    for (let g = 0; g < GROUPS; g++) counts[g + 1] += counts[g];
    const cursor = counts.slice(0, GROUPS);
    for (let e = 0; e < m; e++) {
      if (bucket[e] === 0xffff) continue;
      grouped[cursor[bucket[e]]++] = e;
    }

    // Colour strings built once per bucket rather than once per edge.
    const table = new Array(CB);
    for (let c = 0; c < CB; c++) {
      table[c] = flat
        ? s.edgeFlatColor
        : (s.edgeColorBy === 'source'
            ? colormapCss(s.nodeColormap, CB === 1 ? 0.6 : c / (CB - 1), 1, s.nodeColorReverse)
            : colormapCss(s.edgeColormap, CB === 1 ? 0.6 : c / (CB - 1), 1, s.edgeColorReverse));
    }

    for (let g = 0; g < GROUPS; g++) {
      const from = counts[g], to = counts[g + 1];
      if (from === to) continue;

      ctx.strokeStyle = table[(g / WB) | 0];
      ctx.lineWidth = WB === 1 ? widths[grouped[from]]
                               : loW + ((g % WB) + 0.5) / WB * wRange;
      ctx.beginPath();
      for (let idx = from; idx < to; idx++) {
        const e = grouped[idx];
        const ia = pairs[e * 2], ib = pairs[e * 2 + 1];
        ctx.moveTo(sx[ia], sy[ia]);
        ctx.lineTo(sx[ib], sy[ib]);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  /**
   * Draw the nodes, grouped by colour.
   *
   * Assigning `fillStyle` is not free: the canvas re-parses the colour string
   * every time. At eighteen thousand nodes that alone measured around 12ms a
   * frame — more than the arcs themselves. Quantising colour into a few dozen
   * buckets and drawing each bucket as a single path with one `fillStyle` and
   * one `fill` turns eighteen thousand assignments into a few dozen.
   *
   * Depth order is preserved *within* each bucket, so nearer nodes of the same
   * colour still cover farther ones. Across buckets it is not, which means a
   * distant node of one colour can paint over a nearer node of another. At the
   * size these dots are drawn that is not a difference worth 12ms a frame.
   */
  _nodes(ctx, frame, metrics, s) {
    const n = frame.ids.length;
    const sx = this._sx, sy = this._sy, sk = this._sk, depth = this._sDepth, ok = this._sOk;

    const BUCKETS = 48;
    if (!this._bucket || this._bucket.length < n) {
      this._bucket = new Uint8Array(n);
      this._sorted = new Int32Array(n);
      this._grouped = new Int32Array(n);
    }
    const bucket = this._bucket, sorted = this._sorted, grouped = this._grouped;

    // Colour table, built once instead of a string per node.
    const table = new Array(BUCKETS);
    const constant = s.nodeColorBy === 'constant';
    for (let b = 0; b < BUCKETS; b++) {
      table[b] = colormapCss(s.nodeColormap, constant ? 0.6 : b / (BUCKETS - 1), 1, s.nodeColorReverse);
    }

    for (let i = 0; i < n; i++) {
      const t = constant ? 0.6 : metrics.nodeColorNorm(i);
      const b = (t * (BUCKETS - 1)) | 0;
      bucket[i] = b < 0 ? 0 : (b > BUCKETS - 1 ? BUCKETS - 1 : b);
    }

    // Back to front, so nearer nodes cover farther ones. In 2D every depth is
    // equal and the ordering is skipped entirely.
    //
    // Sorting this by comparator cost 8.2ms at eighteen thousand nodes — the
    // single most expensive thing left in the draw — because a JS comparator
    // is called a quarter of a million times and none of it inlines. Depth
    // only has to be right to within a pixel of overlap, so it is quantised
    // into bins and counting-sorted instead: same order to the eye, 0.16ms.
    // The pass is stable, so nodes sharing a bin keep their frame order.
    let order = sorted;
    if (!this.mode3D) {
      for (let i = 0; i < n; i++) sorted[i] = i;
    } else {
      const BINS = 512;
      if (!this._depthBins) this._depthBins = new Int32Array(BINS + 1);
      const bins = this._depthBins;

      let lo = Infinity, hi = -Infinity;
      for (let i = 0; i < n; i++) {
        const d = depth[i];
        if (d < lo) lo = d;
        if (d > hi) hi = d;
      }
      const span = (hi - lo) || 1;
      const scale = (BINS - 1) / span;

      bins.fill(0);
      for (let i = 0; i < n; i++) bins[(((depth[i] - lo) * scale) | 0) + 1]++;
      for (let b = 0; b < BINS; b++) bins[b + 1] += bins[b];
      for (let i = 0; i < n; i++) sorted[bins[((depth[i] - lo) * scale) | 0]++] = i;
    }

    // Counting sort into colour groups. Walking `order` keeps it stable, so
    // depth order survives inside each bucket.
    const counts = new Int32Array(BUCKETS + 1);
    for (let i = 0; i < n; i++) counts[bucket[i] + 1]++;
    for (let b = 0; b < BUCKETS; b++) counts[b + 1] += counts[b];
    const cursor = counts.slice(0, BUCKETS);
    for (let idx = 0; idx < n; idx++) {
      const i = order[idx];
      grouped[cursor[bucket[i]]++] = i;
    }

    const w = this.cssWidth, h = this.cssHeight;
    const outline = s.nodeOutline;

    // Radius is wanted twice — once for the glow, once for the node — and it
    // is the same number both times.
    if (!this._sRadius || this._sRadius.length < n) this._sRadius = new Float64Array(n);
    const radii = this._sRadius;
    for (let i = 0; i < n; i++) {
      let r = s.nodeSizeMin + (s.nodeSizeMax - s.nodeSizeMin) * metrics.nodeSizeNorm(i);
      if (this.mode3D) r *= sk[i];
      radii[i] = r;
    }

    const visible = (i) => ok[i] && sx[i] >= -50 && sy[i] >= -50
                        && sx[i] <= w + 50 && sy[i] <= h + 50;

    if (s.nodeGlow) this._glow(ctx, s, table, counts, grouped, radii, visible);

    ctx.globalAlpha = s.nodeAlpha;

    for (let b = 0; b < BUCKETS; b++) {
      const from = counts[b], to = counts[b + 1];
      if (from === to) continue;

      ctx.fillStyle = table[b];
      ctx.beginPath();
      let drew = false;

      for (let idx = from; idx < to; idx++) {
        const i = grouped[idx];
        if (!visible(i)) continue;

        const radius = radii[i];
        if (radius < 0.2) continue;

        // moveTo first, or the arcs are joined by stray lines.
        const x = sx[i], y = sy[i];
        ctx.moveTo(x + radius, y);
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        drew = true;
      }

      if (drew) {
        ctx.fill();
        if (outline) {
          ctx.globalAlpha = Math.min(1, s.nodeAlpha * s.nodeOutlineAlpha + 0.2);
          ctx.strokeStyle = s.nodeOutlineColor;
          ctx.lineWidth = s.nodeOutlineWidth;
          ctx.stroke();
          ctx.globalAlpha = s.nodeAlpha;
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  /**
   * A halo under each node.
   *
   * Canvas shadows would be the obvious way and are far too slow to use per
   * node at this scale. Instead each node gets a pair of oversized translucent
   * discs drawn additively: where they overlap they accumulate, which is what
   * makes a dense cluster read as brighter than a lone node — the thing that
   * actually looks like light.
   *
   * Additive blending means the glow only shows against a dark background. On
   * a white one it has nothing to add to and quietly does nothing.
   *
   * Colouring by the outline gives one fill for every node on screen, so the
   * whole pass is a single path. Colouring by the node reuses the buckets the
   * caller already sorted, costing one fill per bucket rather than per node.
   */
  _glow(ctx, s, table, counts, grouped, radii, visible) {
    const sx = this._sx, sy = this._sy;
    const BUCKETS = table.length;
    const byNode = s.nodeGlowColorBy === 'node';
    const size = Math.max(1.05, s.nodeGlowSize);

    ctx.save();
    ctx.globalCompositeOperation = 'lighter';

    // Two rings rather than one, so the halo fades outward instead of ending
    // at a hard edge. The inner ring carries most of the weight.
    const rings = [[size, 0.45], [1 + (size - 1) * 0.5, 0.55]];

    for (const [scale, share] of rings) {
      ctx.globalAlpha = s.nodeGlowStrength * share;

      if (byNode) {
        for (let b = 0; b < BUCKETS; b++) {
          const from = counts[b], to = counts[b + 1];
          if (from === to) continue;
          ctx.fillStyle = table[b];
          ctx.beginPath();
          let drew = false;
          for (let idx = from; idx < to; idx++) {
            const i = grouped[idx];
            if (!visible(i) || radii[i] < 0.2) continue;
            const r = radii[i] * scale;
            ctx.moveTo(sx[i] + r, sy[i]);
            ctx.arc(sx[i], sy[i], r, 0, Math.PI * 2);
            drew = true;
          }
          if (drew) ctx.fill();
        }
      } else {
        ctx.fillStyle = s.nodeOutlineColor;
        ctx.beginPath();
        let drew = false;
        for (let idx = 0; idx < counts[BUCKETS]; idx++) {
          const i = grouped[idx];
          if (!visible(i) || radii[i] < 0.2) continue;
          const r = radii[i] * scale;
          ctx.moveTo(sx[i] + r, sy[i]);
          ctx.arc(sx[i], sy[i], r, 0, Math.PI * 2);
          drew = true;
        }
        if (drew) ctx.fill();
      }
    }

    ctx.restore();
  }

  /**
   * One labelled colour strip: what the map means and the ends of its range.
   */
  _legendBlock(ctx, x, y, label, rangeText, colormap, reverse) {
    const w = 150, h = 10;

    ctx.globalAlpha = 0.92;
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(x - 8, y - 20, w + 16, h + 38);

    drawColormapStrip(ctx, x, y, w, h, colormap, reverse);

    ctx.fillStyle = '#e6edf3';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(label, x, y - 7);

    ctx.font = '10px system-ui, sans-serif';
    ctx.fillStyle = '#9fb0c0';
    const [lo, hi] = rangeText;
    ctx.fillText(lo, x, y + h + 11);
    ctx.fillText(hi, x + w - ctx.measureText(hi).width, y + h + 11);
  }

  /**
   * The colour keys, stacked up from the bottom left.
   *
   * The edge key is only worth drawing when edge colour is actually reading
   * something: a flat colour has no range to show, and inheriting the node
   * colour would just repeat the key already above it.
   */
  _legend(ctx, metrics, s) {
    const blocks = [];
    if (s.showLegend) {
      blocks.push([metrics.colorLabel, metrics.colorRangeText,
                   s.nodeColormap, s.nodeColorReverse]);
    }
    if (s.showEdgeLegend && s.edgeShow
        && s.edgeColorBy !== 'constant' && s.edgeColorBy !== 'source') {
      blocks.push([metrics.edgeColorLabel, metrics.edgeColorRangeText,
                   s.edgeColormap, s.edgeColorReverse]);
    }
    if (!blocks.length) return;

    ctx.save();
    // Bottom-most first, each one 48px above the last.
    let y = this.cssHeight - 40;
    for (const [label, range, colormap, reverse] of blocks) {
      this._legendBlock(ctx, 14, y, label, range, colormap, reverse);
      y -= 48;
    }
    ctx.restore();
  }

  /** Nearest node to a screen point, for hover. Returns an index or -1. */
  pick(frame, layout, px, py, maxPixels = 12) {
    if (!frame || !this._sOk) return -1;
    let best = -1, bestDist = maxPixels * maxPixels;
    const sx = this._sx, sy = this._sy, ok = this._sOk;

    for (let i = 0; i < frame.ids.length; i++) {
      if (!ok[i]) continue;
      const dx = sx[i] - px, dy = sy[i] - py;
      const dSq = dx * dx + dy * dy;
      if (dSq < bestDist) { bestDist = dSq; best = i; }
    }
    return best;
  }
}
