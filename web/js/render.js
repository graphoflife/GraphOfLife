/*
 * Canvas renderer for a single frame.
 *
 * Everything visual is driven by a settings object: which quantity drives node
 * colour, which drives node size, the same for edges, background style, and
 * opacities. The renderer only maps numbers to pixels — it never decides what
 * a value means.
 */
class GraphRenderer {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.view = { scale: 1, offsetX: 0, offsetY: 0 };
    this.dpr = window.devicePixelRatio || 1;
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    this.dpr = window.devicePixelRatio || 1;
    this.canvas.width = Math.max(1, Math.floor(rect.width * this.dpr));
    this.canvas.height = Math.max(1, Math.floor(rect.height * this.dpr));
    this.cssWidth = rect.width;
    this.cssHeight = rect.height;
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

  toScreen(p) {
    return {
      x: p.x * this.view.scale + this.view.offsetX,
      y: p.y * this.view.scale + this.view.offsetY
    };
  }

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
      if (settings.edgeShow) this._edges(ctx, frame, metrics, layout, settings);
      this._nodes(ctx, frame, metrics, layout, settings);
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

  _edges(ctx, frame, metrics, layout, s) {
    const edges = frame.edges;
    const flat = s.edgeColorBy === 'constant';

    ctx.globalAlpha = s.edgeAlpha;
    if (flat) {
      ctx.strokeStyle = s.edgeFlatColor;
      ctx.lineWidth = s.edgeWidthMin;
    }

    // A single path is far cheaper than per-edge strokes, so the flat/constant
    // combination — the common case at scale — gets a fast path.
    const uniformWidth = s.edgeWidthBy === 'constant';
    if (flat && uniformWidth) {
      ctx.beginPath();
      for (const [a, b] of edges) {
        const pa = layout.pos.get(a), pb = layout.pos.get(b);
        if (!pa || !pb) continue;
        const sa = this.toScreen(pa), sb = this.toScreen(pb);
        ctx.moveTo(sa.x, sa.y);
        ctx.lineTo(sb.x, sb.y);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
      return;
    }

    for (const [a, b] of edges) {
      const pa = layout.pos.get(a), pb = layout.pos.get(b);
      if (!pa || !pb) continue;

      const t = metrics.edgeColorNorm(a, b);
      ctx.strokeStyle = flat
        ? s.edgeFlatColor
        : (s.edgeColorBy === 'source'
            ? metrics.nodeColorCss(a, 1)
            : colormapCss(s.edgeColormap, t, 1, false));

      const wNorm = metrics.edgeWidthNorm(a, b);
      ctx.lineWidth = s.edgeWidthMin + (s.edgeWidthMax - s.edgeWidthMin) * wNorm;

      const sa = this.toScreen(pa), sb = this.toScreen(pb);
      ctx.beginPath();
      ctx.moveTo(sa.x, sa.y);
      ctx.lineTo(sb.x, sb.y);
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  _nodes(ctx, frame, metrics, layout, s) {
    const ids = frame.ids;
    ctx.globalAlpha = s.nodeAlpha;

    for (let i = 0; i < ids.length; i++) {
      const p = layout.pos.get(ids[i]);
      if (!p) continue;

      const screen = this.toScreen(p);
      // Skip anything comfortably off-screen.
      if (screen.x < -50 || screen.y < -50 ||
          screen.x > this.cssWidth + 50 || screen.y > this.cssHeight + 50) continue;

      const radius = s.nodeSizeMin + (s.nodeSizeMax - s.nodeSizeMin) * metrics.nodeSizeNorm(i);
      ctx.fillStyle = metrics.nodeColorCssByIndex(i, 1);
      ctx.beginPath();
      ctx.arc(screen.x, screen.y, radius, 0, Math.PI * 2);
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

    for (let i = 0; i < frame.ids.length; i++) {
      const p = layout.pos.get(frame.ids[i]);
      if (!p) continue;
      const screen = this.toScreen(p);
      const dx = screen.x - sx, dy = screen.y - sy;
      const dSq = dx * dx + dy * dy;
      if (dSq < bestDist) { bestDist = dSq; best = i; }
    }
    return best;
  }
}
