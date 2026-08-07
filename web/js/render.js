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
    this.view = { scale: 1, offsetX: 0, offsetY: 0 };
    this.dpr = window.devicePixelRatio || 1;

    this.mode3D = false;
    this.yaw = 0.6;
    this.pitch = -0.35;
    // Distance of the eye from the scene centre, in world units. Larger means
    // a flatter, more orthographic look.
    this.cameraDistance = 900;

    this._depthOrder = [];
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
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
  fitToContent(layout, padding = 0.015) {
    if (!layout || !layout.ids.length || !this.cssWidth) return;

    const passes = this.mode3D ? 2 : 1;
    for (let pass = 0; pass < passes; pass++) {
      const box = this._projectedBounds(layout);
      if (!box) return;

      const availableW = this.cssWidth * (1 - padding * 2);
      const availableH = this.cssHeight * (1 - padding * 2);
      const factor = Math.min(availableW / box.width, availableH / box.height);
      if (!Number.isFinite(factor) || factor <= 0) return;

      this.view.scale = Math.max(0.02, Math.min(60, this.view.scale * factor));

      // Scaling moves everything, so recentre against the new projection.
      const after = this._projectedBounds(layout);
      if (!after) return;
      this.view.offsetX += this.cssWidth / 2 - after.centerX;
      this.view.offsetY += this.cssHeight / 2 - after.centerY;
    }
  }

  /** Screen-space extent of every node under the current camera. */
  _projectedBounds(layout) {
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

    for (const id of layout.ids) {
      const p = layout.pos.get(id);
      if (!p) continue;
      const s = this.project(p);
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
    this.view.scale = Math.max(0.02, Math.min(60, this.view.scale * factor));
    const after = this.toWorld(sx, sy);
    this.view.offsetX += (after.x - before.x) * this.view.scale;
    this.view.offsetY += (after.y - before.y) * this.view.scale;
  }

  pan(dx, dy) {
    this.view.offsetX += dx;
    this.view.offsetY += dy;
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
      const screen = new Map();
      for (const id of frame.ids) {
        const p = layout.pos.get(id);
        if (p) screen.set(id, this.project(p));
      }
      this._screen = screen;

      if (settings.edgeShow) this._edges(ctx, frame, metrics, screen, settings);
      this._nodes(ctx, frame, metrics, screen, settings);
      if (settings.showLegend) this._legend(ctx, metrics, settings);
    }

    ctx.restore();
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

  _edges(ctx, frame, metrics, screen, s) {
    const edges = frame.edges;
    const flat = s.edgeColorBy === 'constant';
    const uniformWidth = s.edgeWidthBy === 'constant';

    ctx.globalAlpha = s.edgeAlpha;

    // A single path is far cheaper than per-edge strokes, so the flat/constant
    // combination — the common case at scale — gets a fast path.
    if (flat && uniformWidth) {
      ctx.strokeStyle = s.edgeFlatColor;
      ctx.lineWidth = s.edgeWidthMin;
      ctx.beginPath();
      for (const [a, b] of edges) {
        const sa = screen.get(a), sb = screen.get(b);
        if (!sa || !sb) continue;
        ctx.moveTo(sa.x, sa.y);
        ctx.lineTo(sb.x, sb.y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      return;
    }

    for (const [a, b] of edges) {
      const sa = screen.get(a), sb = screen.get(b);
      if (!sa || !sb) continue;

      const t = metrics.edgeColorNorm(a, b);
      ctx.strokeStyle = flat
        ? s.edgeFlatColor
        : (s.edgeColorBy === 'source'
            ? metrics.nodeColorCss(a, 1)
            : colormapCss(s.edgeColormap, t, 1, false));

      const wNorm = metrics.edgeWidthNorm(a, b);
      let width = s.edgeWidthMin + (s.edgeWidthMax - s.edgeWidthMin) * wNorm;
      if (this.mode3D) width *= (sa.k + sb.k) / 2;

      ctx.lineWidth = Math.max(0.05, width);
      ctx.beginPath();
      ctx.moveTo(sa.x, sa.y);
      ctx.lineTo(sb.x, sb.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  _nodes(ctx, frame, metrics, screen, s) {
    const ids = frame.ids;
    ctx.globalAlpha = s.nodeAlpha;

    // Back to front, so nearer nodes cover farther ones. In 2D every depth is
    // equal and the sort is skipped entirely.
    let order = this._depthOrder;
    if (order.length !== ids.length) {
      order = this._depthOrder = new Array(ids.length);
    }
    for (let i = 0; i < ids.length; i++) order[i] = i;

    if (this.mode3D) {
      order.sort((i, j) => {
        const a = screen.get(ids[i]), b = screen.get(ids[j]);
        return (a ? a.depth : 0) - (b ? b.depth : 0);
      });
    }

    for (const i of order) {
      const point = screen.get(ids[i]);
      if (!point) continue;

      // Skip anything comfortably off-screen.
      if (point.x < -50 || point.y < -50 ||
          point.x > this.cssWidth + 50 || point.y > this.cssHeight + 50) continue;

      let radius = s.nodeSizeMin + (s.nodeSizeMax - s.nodeSizeMin) * metrics.nodeSizeNorm(i);
      if (this.mode3D) radius *= point.k;
      if (radius < 0.2) continue;

      ctx.fillStyle = metrics.nodeColorCssByIndex(i, 1);
      ctx.beginPath();
      ctx.arc(point.x, point.y, radius, 0, Math.PI * 2);
      ctx.fill();

      if (s.nodeOutline && radius > 2) {
        ctx.globalAlpha = Math.min(1, s.nodeAlpha + 0.2);
        ctx.strokeStyle = 'rgba(0,0,0,0.55)';
        ctx.lineWidth = 0.6;
        ctx.stroke();
        ctx.globalAlpha = s.nodeAlpha;
      }
    }
    ctx.globalAlpha = 1;
  }

  _legend(ctx, metrics, s) {
    const w = 150, h = 10;
    const x = 14, y = this.cssHeight - 40;

    ctx.save();
    ctx.globalAlpha = 0.92;
    ctx.fillStyle = 'rgba(0,0,0,0.45)';
    ctx.fillRect(x - 8, y - 20, w + 16, h + 38);

    drawColormapStrip(ctx, x, y, w, h, s.nodeColormap, s.nodeColorReverse);

    ctx.fillStyle = '#e6edf3';
    ctx.font = '11px system-ui, sans-serif';
    ctx.fillText(metrics.colorLabel, x, y - 7);

    ctx.font = '10px system-ui, sans-serif';
    ctx.fillStyle = '#9fb0c0';
    const [lo, hi] = metrics.colorRangeText;
    ctx.fillText(lo, x, y + h + 11);
    const hiWidth = ctx.measureText(hi).width;
    ctx.fillText(hi, x + w - hiWidth, y + h + 11);
    ctx.restore();
  }

  /** Nearest node to a screen point, for hover. Returns an index or -1. */
  pick(frame, layout, sx, sy, maxPixels = 12) {
    if (!frame) return -1;
    let best = -1, bestDist = maxPixels * maxPixels;
    const screen = this._screen;

    for (let i = 0; i < frame.ids.length; i++) {
      const point = screen ? screen.get(frame.ids[i]) : null;
      const projected = point || (() => {
        const p = layout.pos.get(frame.ids[i]);
        return p ? this.project(p) : null;
      })();
      if (!projected) continue;

      const dx = projected.x - sx, dy = projected.y - sy;
      const dSq = dx * dx + dy * dy;
      if (dSq < bestDist) { bestDist = dSq; best = i; }
    }
    return best;
  }
}
