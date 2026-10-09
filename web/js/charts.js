/*
 * The charts the pages draw on canvases: a histogram, a heatmap, a trajectory
 * and a time series, on the axes, legends and colour bars they share.
 *
 * They lived at the end of stats.js, which had grown past fifteen hundred
 * lines by holding both a frame's numbers and every way of drawing them.
 */

// --------------------------------------------------------------------------
// Charts
// --------------------------------------------------------------------------

/** Shared canvas setup: size to the element's box in device pixels. */
function _prepareCanvas(canvas, pad = null) {
  const ctx = canvas.getContext('2d');
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();

  canvas.width = Math.max(1, Math.floor(rect.width * dpr));
  canvas.height = Math.max(1, Math.floor(rect.height * dpr));
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!pad) return { ctx, w: rect.width, h: rect.height, dpr, outer: rect };

  // Translating rather than threading an offset through every coordinate: the
  // plotting arithmetic in each chart then works unchanged inside the smaller
  // rectangle, and the furniture around it is drawn by resetting the transform.
  ctx.translate(pad.left, pad.top);
  return {
    ctx, dpr, pad, outer: rect,
    w: Math.max(1, rect.width - pad.left - pad.right),
    h: Math.max(1, rect.height - pad.top - pad.bottom)
  };
}

/**
 * How much room the furniture round a chart needs.
 *
 * Nothing is reserved for a part that was not asked for, so a chart with no
 * title loses no height to one — which is what keeps the Viewer's compact
 * strip exactly as it was while the Diagrams tab gets a full-dress chart.
 */
function _chromePad(chrome) {
  if (!chrome) return null;
  const rows = chrome.legend ? chrome.legend.length : 0;
  return {
    left: chrome.yLabel || chrome.ticks ? 56 : 8,
    right: 14,
    top: chrome.title ? 26 : 8,
    bottom: (chrome.ticks ? 30 : 10) + (chrome.xLabel ? 14 : 0) + rows * 14
  };
}

/** A number short enough to sit under a tick without colliding with the next. */
function _short(v) {
  const a = Math.abs(v);
  if (a >= 1000) return Math.round(v).toLocaleString('en-US');
  if (a >= 10) return v.toFixed(1);
  if (a >= 0.01) return v.toFixed(3);
  if (a === 0) return '0';
  return v.toExponential(1);
}


/** Round numbers to rule an axis at — the same stepping the history chart uses. */
function _axisTicks(lo, hi, target = 5) {
  if (!(hi > lo)) return [lo];
  const rough = (hi - lo) / Math.max(1, target);
  const magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
  const scaled = rough / magnitude;
  const step = magnitude *
    (scaled <= 1 ? 1 : scaled <= 2 ? 2 : scaled <= 2.5 ? 2.5 : scaled <= 5 ? 5 : 10);
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-6; v += step) {
    out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  }
  return out;
}

/**
 * Grid, ticks and any constant lines, in the chart's own coordinates.
 *
 * Called once a chart knows its domain and before it draws its data, so the
 * grid sits behind rather than over — a heatmap with gridlines on top of the
 * cells is harder to read, not easier.
 */
/**
 * How many ticks a plot's axes have room for, rather than a fixed five. The
 * same chart is drawn at 300px in the Research pane and at 1600px in a saved
 * image; five is crowded at one end and sparse at the other. A label needs
 * roughly 90px across and a row 48px down before they start touching.
 */
function _tickCounts(w, h) {
  return {
    across: Math.max(2, Math.min(8, Math.round(w / 90))),
    down: Math.max(2, Math.min(6, Math.round(h / 48)))
  };
}

function _axes(ctx, w, h, spec) {
  // An axis may bring its own `ticks`, in its own units; otherwise round ones
  // are chosen. `pad` is the margin the chrome reserved outside the plot. Tick labels are
  // allowed to use it, and are clamped to it, so an edge tick stays legible
  // instead of being painted off the canvas.
  const { x, y, grid = true, guides = [],
          pad = { left: 0, right: 0 } } = spec;
  ctx.save();
  ctx.font = Ink.font(10);
  ctx.lineWidth = 1;

  const place = (axis, v) => axis.lo === axis.hi ? 0.5 : (v - axis.lo) / (axis.hi - axis.lo);
  const { across, down } = _tickCounts(w, h);

  if (x) {
    for (const tick of x.ticks || _axisTicks(x.lo, x.hi, across)) {
      const at = Math.round(place(x, tick) * w) + 0.5;
      if (grid) {
        ctx.strokeStyle = Ink.of('grid');
        ctx.beginPath(); ctx.moveTo(at, 0); ctx.lineTo(at, h); ctx.stroke();
      }
      // A mark on the axis itself, so a tick is still located when the grid is
      // switched off and the number below it has nothing pointing at it.
      ctx.strokeStyle = Ink.of('tick');
      ctx.beginPath(); ctx.moveTo(at, h); ctx.lineTo(at, h + 4); ctx.stroke();
      ctx.fillStyle = Ink.of('label');
      const text = x.format ? x.format(tick) : String(tick);
      // Held inside the canvas rather than centred and allowed to run off it.
      // The last tick sits on the right edge, so a centred label is half
      // outside and the browser simply does not paint it — the axis then ends
      // without saying what it ends at.
      const width = ctx.measureText(text).width;
      const left = Math.min(Math.max(at - width / 2, -pad.left + 1),
                            w + pad.right - width - 1);
      ctx.fillText(text, left, h + 14);
    }
  }
  if (y) {
    for (const tick of y.ticks || _axisTicks(y.lo, y.hi, down)) {
      const at = Math.round(h - place(y, tick) * h) + 0.5;
      if (grid) {
        ctx.strokeStyle = Ink.of('grid');
        ctx.beginPath(); ctx.moveTo(0, at); ctx.lineTo(w, at); ctx.stroke();
      }
      ctx.strokeStyle = Ink.of('tick');
      ctx.beginPath(); ctx.moveTo(-4, at); ctx.lineTo(0, at); ctx.stroke();
      ctx.fillStyle = Ink.of('label');
      const text = y.format ? y.format(tick) : String(tick);
      // Nudged in from the edges for the same reason, so the top and bottom
      // labels are not clipped by the plot's own boundary.
      const top = Math.min(Math.max(at + 3, 8), h - 1);
      ctx.fillText(text, -ctx.measureText(text).width - 7, top);
    }
  }

  // A line the reader put there: a threshold, a target, a value worth
  // comparing against. Dashed, so it cannot be mistaken for data.
  for (const guide of guides) {
    const axis = guide.axis === 'y' ? y : x;
    if (!axis || !Number.isFinite(guide.at)) continue;
    const t = place(axis, guide.at);
    if (t < -0.02 || t > 1.02) continue;
    ctx.save();
    ctx.setLineDash([5, 4]);
    ctx.strokeStyle = guide.colour || Ink.of('label');
    ctx.beginPath();
    if (guide.axis === 'y') {
      const at = Math.round(h - t * h) + 0.5;
      ctx.moveTo(0, at); ctx.lineTo(w, at);
    } else {
      const at = Math.round(t * w) + 0.5;
      ctx.moveTo(at, 0); ctx.lineTo(at, h);
    }
    ctx.stroke();
    ctx.restore();
  }

  ctx.strokeStyle = Ink.of('axis');
  ctx.strokeRect(0.5, 0.5, w - 1, h - 1);
  ctx.restore();
}

/**
 * Title, axis names and legend, outside the plotting area.
 *
 * Drawn into the canvas rather than laid beside it in HTML, because a saved
 * image has to carry them: a legend that lives in the page is not in the PNG,
 * and a chart of eight unnamed lines is not worth saving.
 */
function _chrome(ctx, pad, outer, chrome) {
  if (!chrome) return;
  ctx.save();
  ctx.setTransform(ctx.getTransform().a, 0, 0, ctx.getTransform().d, 0, 0);
  const w = outer.width, h = outer.height;

  if (chrome.title) {
    ctx.fillStyle = Ink.of('text');
    ctx.font = Ink.font(13, 600);
    ctx.fillText(chrome.title, pad.left, 17);
  }
  ctx.font = Ink.font(11);
  ctx.fillStyle = Ink.of('label');

  if (chrome.xLabel) {
    const width = ctx.measureText(chrome.xLabel).width;
    const rows = chrome.legend ? chrome.legend.length : 0;
    ctx.fillText(chrome.xLabel, pad.left + (w - pad.left - pad.right - width) / 2,
                 h - 6 - rows * 14);
  }
  if (chrome.yLabel) {
    ctx.save();
    ctx.translate(12, pad.top + (h - pad.top - pad.bottom) / 2);
    ctx.rotate(-Math.PI / 2);
    const width = ctx.measureText(chrome.yLabel).width;
    ctx.fillText(chrome.yLabel, -width / 2, 0);
    ctx.restore();
  }
  for (const [i, entry] of (chrome.legend || []).entries()) {
    const y = h - 6 - (chrome.legend.length - 1 - i) * 14;
    ctx.fillStyle = entry.colour;
    ctx.fillRect(pad.left, y - 7, 14, 3);
    ctx.fillStyle = Ink.of('label');
    ctx.fillText(entry.label, pad.left + 20, y);
  }
  ctx.restore();
}

/**
 * Bin edges that no integer can fall between.
 *
 * A log axis spreads small numbers apart and squeezes large ones together.
 * Tokens and degrees are whole numbers, so near the origin two neighbouring
 * values can sit more than a bin apart — with thirty-four bins over 1 to 1200,
 * the step from one token to two is 2.15 bins wide. The bin in between cannot
 * receive anything, because there is no such thing as one and a half tokens,
 * and it shows up as a blank column running through the plot.
 *
 * So: start from evenly spaced edges, then drop any edge that would leave a
 * bin containing no whole number, merging it into the one after. Where values
 * are dense the edges are untouched and the resolution is unchanged; near the
 * origin a few bins come out wider, which is exactly the width the data
 * actually has there.
 *
 * Only for whole numbers. Anything continuous can land anywhere, so every bin
 * is reachable and the even spacing is left alone.
 */
function _binEdges([lo, hi], bins, { integral, unmap }) {
  const even = [];
  for (let i = 0; i <= bins; i++) even.push(lo + (hi - lo) * (i / bins));
  if (!integral) return even;

  const holdsWholeNumber = (from, to) => {
    const a = unmap(from), b = unmap(to);
    // A whole number lies in [a, b) if rounding a up gets there before b does.
    return Math.floor(b - 1e-9) >= Math.ceil(a - 1e-9);
  };

  const kept = [even[0]];
  for (let i = 1; i < bins; i++) {
    if (holdsWholeNumber(kept[kept.length - 1], even[i])) kept.push(even[i]);
  }
  kept.push(even[bins]);   // the axis has to end where the data does
  return kept;
}

/** Whether every value is a whole number, so bins between them are impossible. */
function _allWholeNumbers(values) {
  for (const v of values) {
    if (!Number.isFinite(v) || !Number.isInteger(v)) return false;
  }
  return true;
}

/** Which bin a value falls in, given edges that are not evenly spaced. */
function _binOf(value, edges) {
  let low = 0, high = edges.length - 1;
  while (low < high - 1) {
    const mid = (low + high) >> 1;
    if (value < edges[mid]) high = mid; else low = mid;
  }
  return Math.min(low, edges.length - 2);
}

function _noData(ctx, w, h, message = 'no data') {
  ctx.fillStyle = Ink.of('dim');
  ctx.font = Ink.font(11);
  ctx.fillText(message, 8, h / 2);
}

/**
 * Draw a bar histogram of one metric.
 *
 * Values arrive raw; `logScale` is applied here rather than by the caller so
 * the axis can be labelled in the units the reader actually chose. `signed`
 * keeps the direction of quantities that run either way, which is what lets
 * token change and curvature be read on a log scale at all.
 */
function drawHistogram(canvas, values, options = {}) {
  const { bins = 32, colormap = 'viridis', reverse = false,
          logScale = false, logCount = false, signed = false, chrome = null,
          format = v => Math.round(v).toLocaleString('en-US') } = options;

  const pad = _chromePad(chrome);
  const { ctx, w, h, outer } = _prepareCanvas(canvas, pad);
  if (!values || !values.length) { _noData(ctx, w, h); return; }

  // Values can be missing — a "before the phase" quantity has none for a node
  // that did not exist yet. Those are dropped rather than counted as zero.
  const mapped = [];
  let skipped = 0;
  for (const raw of values) {
    if (Number.isNaN(raw)) { skipped++; continue; }
    mapped.push(logScale ? Metrics.applyLog(raw, signed) : raw);
  }
  if (!mapped.length) { _noData(ctx, w, h, 'no values for this frame'); return; }

  let lo = Infinity, hi = -Infinity;
  for (const v of mapped) { if (v < lo) lo = v; if (v > hi) hi = v; }
  if (!Number.isFinite(lo)) { _noData(ctx, w, h); return; }
  if (hi - lo < 1e-9) hi = lo + 1;

  // The same reason the heatmap needs them: on a log axis, whole numbers near
  // the origin are further apart than one bin, and the bins between them can
  // never be filled.
  const edges = _binEdges([lo, hi], bins, {
    integral: _allWholeNumbers(values),
    unmap: v => (logScale ? Metrics.undoLog(v, signed) : v)
  });
  const count = edges.length - 1;

  const counts = new Array(count).fill(0);
  for (const v of mapped) counts[_binOf(v, edges)]++;
  const peak = Math.max(...counts) || 1;

  // With furniture the axes carry the numbers, so the chart keeps no room for
  // its own footer line.
  const padBottom = chrome ? 0 : 16, padTop = chrome ? 0 : 6;
  const plotH = h - padBottom - padTop;
  const at = v => (v - lo) / (hi - lo) * w;

  if (chrome) {
    const back = v => logScale ? Metrics.undoLog(v, signed) : v;
    _axes(ctx, w, h, {
      x: { lo, hi, format: v => format(back(v)) },
      y: { lo: 0, hi: Math.max(...counts) || 1,
           format: v => Math.round(v).toLocaleString('en-US') },
      grid: chrome.grid !== false, guides: chrome.guides || [], pad
    });
  }

  // A log count axis lets a long tail of rare values stay visible next to a
  // spike that would otherwise flatten everything else to nothing.
  const barFraction = c => logCount
    ? (c > 0 ? Math.log1p(c) / Math.log1p(peak) : 0)
    : c / peak;

  for (let i = 0; i < count; i++) {
    const barH = barFraction(counts[i]) * plotH;
    // Coloured by where the bar sits on the axis rather than by its index, so
    // a merged wider bin keeps the colour its position deserves.
    const left = at(edges[i]), right = at(edges[i + 1]);
    const shade = count > 1 ? (left / Math.max(1, w)) : 0.5;
    ctx.fillStyle = colormapCss(colormap, Math.min(1, shade), 0.92, reverse);
    ctx.fillRect(left, padTop + plotH - barH,
                 Math.max(1, right - left - 1), barH);
  }

  if (chrome) { _chrome(ctx, pad, outer, chrome); return; }

  const back = v => logScale ? Metrics.undoLog(v, signed) : v;
  ctx.fillStyle = Ink.of('label');
  ctx.font = Ink.font(10);
  ctx.fillText(format(back(lo)), 2, h - 4);
  const hiText = format(back(hi));
  ctx.fillText(hiText, w - ctx.measureText(hiText).width - 2, h - 4);
  const peakText = `peak ${peak}${logCount ? ' \u00b7 log' : ''}`
    + (skipped ? ` \u00b7 ${skipped.toLocaleString('en-US')} without a value` : '');
  ctx.fillText(peakText, (w - ctx.measureText(peakText).width) / 2, h - 4);
}

/**
 * Draw a two-dimensional binned heatmap of one metric against another.
 *
 * Each item — a node, or an edge — contributes one point, so the two arrays
 * must describe the same things in the same order. That is why the caller is
 * required to keep both axes in one domain: pairing a node value with an edge
 * value would count things that have no correspondence at all.
 *
 * Cell colour is the count in that bin, which is nearly always the quantity
 * with the widest spread on the chart: a handful of cells hold most of the
 * population. `logCount` is usually what makes the rest of the grid visible.
 */
function drawHeatmap(canvas, xs, ys, options = {}) {
  const { binsX = 34, binsY = 22, colormap = 'viridis', reverse = false,
          logX = false, logY = false, logCount = true,
          signedX = false, signedY = false,
          chrome = null,
          formatX = v => Math.round(v).toLocaleString('en-US'),
          formatY = v => Math.round(v).toLocaleString('en-US'),
          message = null } = options;

  const pad = _chromePad(chrome);
  const { ctx, w, h, outer } = _prepareCanvas(canvas, pad);
  if (message) { _noData(ctx, w, h, message); return; }
  if (!xs || !ys || !xs.length || xs.length !== ys.length) { _noData(ctx, w, h); return; }

  // A point needs a value on both axes. Where either is missing there is
  // nothing to plot it against, so it is dropped — which is why the two
  // quantities can have different counts and still be compared honestly.
  const mx = [], my = [];
  let dropped = 0;
  for (let i = 0; i < xs.length; i++) {
    const a = xs[i], b = ys[i];
    if (Number.isNaN(a) || Number.isNaN(b)) { dropped++; continue; }
    mx.push(logX ? Metrics.applyLog(a, signedX) : a);
    my.push(logY ? Metrics.applyLog(b, signedY) : b);
  }
  if (!mx.length) { _noData(ctx, w, h, 'nothing has a value on both axes'); return; }

  const extent = (arr) => {
    let lo = Infinity, hi = -Infinity;
    for (const v of arr) { if (v < lo) lo = v; if (v > hi) hi = v; }
    if (!Number.isFinite(lo)) return null;
    return hi - lo < 1e-9 ? [lo, lo + 1] : [lo, hi];
  };
  const ex = extent(mx), ey = extent(my);
  if (!ex || !ey) { _noData(ctx, w, h); return; }

  // With furniture the axes carry the numbers, so the chart keeps no room for
  // its own edge labels.
  const padLeft = chrome ? 0 : 38, padBottom = chrome ? 0 : 15;
  const padTop = chrome ? 0 : 4, padRight = chrome ? 0 : 2;
  const plotW = Math.max(1, w - padLeft - padRight);
  const plotH = Math.max(1, h - padBottom - padTop);

  if (chrome) {
    _axes(ctx, w, h, {
      x: { lo: ex[0], hi: ex[1],
           format: v => formatX(logX ? Metrics.undoLog(v, signedX) : v) },
      y: { lo: ey[0], hi: ey[1],
           format: v => formatY(logY ? Metrics.undoLog(v, signedY) : v) },
      grid: chrome.grid !== false, guides: chrome.guides || [], pad
    });
  }

  // Edges rather than a count, because on a log axis they are not evenly
  // spaced: near the origin whole numbers are further apart than one bin, and
  // the bins between them can never be filled.
  const edgesX = _binEdges(ex, binsX, {
    integral: _allWholeNumbers(xs), unmap: v => (logX ? Metrics.undoLog(v, signedX) : v)
  });
  const edgesY = _binEdges(ey, binsY, {
    integral: _allWholeNumbers(ys), unmap: v => (logY ? Metrics.undoLog(v, signedY) : v)
  });
  const nx = edgesX.length - 1, ny = edgesY.length - 1;

  const counts = new Int32Array(nx * ny);
  for (let i = 0; i < mx.length; i++) {
    counts[_binOf(my[i], edgesY) * nx + _binOf(mx[i], edgesX)]++;
  }
  let peak = 0;
  for (const c of counts) if (c > peak) peak = c;
  if (!peak) { _noData(ctx, w, h); return; }

  const shade = c => logCount ? Math.log1p(c) / Math.log1p(peak) : c / peak;

  // Each cell is drawn at the width its own bin covers, so a wider bin near
  // the origin looks wider — the axis stays a true log axis rather than
  // pretending every bin spans the same amount.
  const atX = v => padLeft + (v - ex[0]) / (ex[1] - ex[0]) * plotW;
  const atY = v => padTop + plotH - (v - ey[0]) / (ey[1] - ey[0]) * plotH;

  // Empty cells stay as background rather than taking the colour map's zero,
  // so "nothing here" reads differently from "the lowest value on the scale".
  for (let by = 0; by < ny; by++) {
    const top = atY(edgesY[by + 1]), bottom = atY(edgesY[by]);
    for (let bx = 0; bx < nx; bx++) {
      const c = counts[by * nx + bx];
      if (!c) continue;
      ctx.fillStyle = colormapCss(colormap, shade(c), 1, reverse);
      const left = atX(edgesX[bx]), right = atX(edgesX[bx + 1]);
      ctx.fillRect(left, top, Math.max(1, Math.ceil(right - left)),
                   Math.max(1, Math.ceil(bottom - top)));
    }
  }

  if (chrome) { _chrome(ctx, pad, outer, chrome); return; }

  const backX = v => logX ? Metrics.undoLog(v, signedX) : v;
  const backY = v => logY ? Metrics.undoLog(v, signedY) : v;

  ctx.fillStyle = Ink.of('label');
  ctx.font = Ink.font(10);

  // y axis: high at the top, low at the bottom of the plot.
  ctx.fillText(formatY(backY(ey[1])), 2, padTop + 8);
  ctx.fillText(formatY(backY(ey[0])), 2, padTop + plotH - 1);

  // x axis, plus what the colour means.
  ctx.fillText(formatX(backX(ex[0])), padLeft, h - 3);
  const hiText = formatX(backX(ex[1]));
  ctx.fillText(hiText, w - ctx.measureText(hiText).width - 2, h - 3);
  const peakText = `${mx.length.toLocaleString('en-US')} paired \u00b7 peak ${peak}`
    + (logCount ? ' \u00b7 log' : '')
    + (dropped ? ` \u00b7 ${dropped.toLocaleString('en-US')} unpaired` : '');
  ctx.fillText(peakText, padLeft + (plotW - ctx.measureText(peakText).width) / 2, h - 3);
}

/**
 * Draw the path two statistics trace against each other over a run.
 *
 * Neither axis is time: each point is one moment, placed by what the two
 * statistics read then, and the colour says when. A run that settles draws a
 * path that wanders into a corner and stays; one that cycles draws a loop;
 * one that never settles keeps moving through fresh colour to the end.
 *
 * Points arrive already ordered in time and already paired — deciding which
 * value belongs with which is the caller's job, since it depends on which
 * phases the two statistics are recorded in.
 */
function drawTrajectory(canvas, points, options = {}) {
  const { colormap = 'viridis', reverse = false,
          logX = false, logY = false,
          xLabel = '', yLabel = '', footer = '', chrome = null,
          message = null } = options;

  const pad = _chromePad(chrome);
  const { ctx, w, h, outer } = _prepareCanvas(canvas, pad);
  if (message) { _noData(ctx, w, h, message); return; }
  if (!points || points.length < 2) {
    _noData(ctx, w, h, 'not enough of the run has both of these yet');
    return;
  }

  const mapX = v => (logX ? Metrics.applyLog(v, v < 0) : v);
  const mapY = v => (logY ? Metrics.applyLog(v, v < 0) : v);

  let loX = Infinity, hiX = -Infinity, loY = Infinity, hiY = -Infinity;
  let loT = Infinity, hiT = -Infinity;
  for (const p of points) {
    const x = mapX(p.x), y = mapY(p.y);
    if (x < loX) loX = x; if (x > hiX) hiX = x;
    if (y < loY) loY = y; if (y > hiY) hiY = y;
    if (p.t < loT) loT = p.t; if (p.t > hiT) hiT = p.t;
  }
  if (!Number.isFinite(loX) || !Number.isFinite(loY)) { _noData(ctx, w, h); return; }
  if (hiX - loX < 1e-12) hiX = loX + 1;
  if (hiY - loY < 1e-12) hiY = loY + 1;
  const spanT = (hiT - loT) || 1;

  // With furniture the axes carry the numbers, so the chart keeps no room for
  // its own.
  const padLeft = chrome ? 0 : 54, padBottom = chrome ? 0 : 30;
  const padTop = chrome ? 0 : 10, padRight = chrome ? 0 : 10;
  const plotW = Math.max(1, w - padLeft - padRight);
  const plotH = Math.max(1, h - padBottom - padTop);
  const px = v => padLeft + (mapX(v) - loX) / (hiX - loX) * plotW;
  const py = v => padTop + plotH - (mapY(v) - loY) / (hiY - loY) * plotH;

  if (chrome) {
    const backX = v => logX ? Metrics.undoLog(v, v < 0) : v;
    const backY = v => logY ? Metrics.undoLog(v, v < 0) : v;
    _axes(ctx, w, h, {
      x: { lo: loX, hi: hiX, format: v => _short(backX(v)) },
      y: { lo: loY, hi: hiY, format: v => _short(backY(v)) },
      grid: chrome.grid !== false, guides: chrome.guides || [], pad
    });
  }

  // Frame, so the path is read against something rather than floating. With
  // chrome, _axes has drawn it already; this used to draw it a second time in
  // a grey of its own.
  if (!chrome) {
    ctx.strokeStyle = Ink.of('axis');
    ctx.lineWidth = 1;
    ctx.strokeRect(padLeft + 0.5, padTop + 0.5, plotW - 1, plotH - 1);
  }

  // One stroke per segment: the colour has to change along the path, and that
  // is the whole point of drawing it this way.
  ctx.lineWidth = 1.6;
  ctx.lineJoin = 'round';
  ctx.lineCap = 'round';
  for (let i = 1; i < points.length; i++) {
    const a = points[i - 1], b = points[i];
    ctx.strokeStyle = colormapCss(colormap, (b.t - loT) / spanT, 0.9, reverse);
    ctx.beginPath();
    ctx.moveTo(px(a.x), py(a.y));
    ctx.lineTo(px(b.x), py(b.y));
    ctx.stroke();
  }

  // Where it started and where it ended, which the colour alone leaves you to
  // work out.
  const first = points[0], last = points[points.length - 1];
  ctx.fillStyle = colormapCss(colormap, 0, 1, reverse);
  ctx.beginPath(); ctx.arc(px(first.x), py(first.y), 3.5, 0, Math.PI * 2); ctx.fill();
  ctx.strokeStyle = Ink.of('text'); ctx.lineWidth = 1; ctx.stroke();
  ctx.fillStyle = colormapCss(colormap, 1, 1, reverse);
  ctx.beginPath(); ctx.arc(px(last.x), py(last.y), 3.5, 0, Math.PI * 2); ctx.fill();
  ctx.stroke();

  const back = (v, log) => (log ? Metrics.undoLog(v, v < 0) : v);

  ctx.font = Ink.font(10);
  ctx.fillStyle = Ink.of('label');
  ctx.fillText(_short(back(hiY, logY)), 2, padTop + 8);
  ctx.fillText(_short(back(loY, logY)), 2, padTop + plotH);
  ctx.fillText(_short(back(loX, logX)), padLeft, h - 17);
  const hiXText = _short(back(hiX, logX));
  ctx.fillText(hiXText, w - ctx.measureText(hiXText).width - 2, h - 17);

  if (!chrome) {
    ctx.fillStyle = Ink.of('label');
    ctx.font = Ink.font(10.5);
    ctx.fillText(`${yLabel} \u2191   vs   ${xLabel} \u2192`, padLeft, padTop - 1);
  }

  // A strip saying which end of the colour map is early and which is late.
  const barW = 90, barH = 7, barX = padLeft, barY = h - 12;
  drawColormapStrip(ctx, barX, barY, barW, barH, colormap, reverse);
  ctx.fillStyle = Ink.of('label');
  ctx.font = Ink.font(9.5);
  ctx.fillText(`iter ${Math.round(loT).toLocaleString('en-US')}`, barX + barW + 6, barY + barH);
  const endText = `\u2192 ${Math.round(hiT).toLocaleString('en-US')}`;
  ctx.fillText(endText, barX + barW + 6 + ctx.measureText(`iter ${Math.round(loT).toLocaleString('en-US')}`).width + 6, barY + barH);
  if (footer) {
    ctx.fillStyle = Ink.of('dim');
    ctx.fillText(footer, w - ctx.measureText(footer).width - 2, barY + barH);
  }
  if (chrome) _chrome(ctx, pad, outer, chrome);
}

/**
 * Statistics through the iterations of a run, one line per track: the
 * Diagrams tab's time series and the Viewer's statistic popup alike, so the
 * two look the same.
 *
 * A track is { points: [{t, v}], mapped, lo, hi, colour, legend, stretch }:
 * its values as they are drawn (`mapped`, on whatever scale the caller chose)
 * and their range, and what the legend says of it. One track gives the
 * vertical axis its own units, read out through `yFormat`, and the axis is
 * widened to the round numbers it labels; several keep a scale each, and the
 * axis says so — a share of each line's own range. A stretched track spans
 * the whole width whatever its length. Guides on axis 'x' cross the
 * iterations; guides on axis 'y' are values, put through `mapY`, and drawn on
 * every track whose range holds them.
 */
function drawTimeline(canvas, tracks, options = {}) {
  const { chrome = null, logX = false, guides = [], mapY = v => v,
          yFormat = v => formatNumber(v), message = null } = options;
  const pad = _chromePad(chrome);
  const { ctx, w, h, outer } = _prepareCanvas(canvas, pad);
  if (message || !tracks.length) {
    ctx.fillStyle = Ink.of('dim');
    ctx.font = Ink.font(13);
    ctx.textAlign = 'center';
    ctx.fillText(message || 'no data', w / 2, h / 2);
    ctx.textAlign = 'left';
    return;
  }

  // Absolute lines share one iteration axis, so a run half as long stops
  // halfway across. A stretched line gets the whole width whatever its
  // length, which is what lets two runs be compared by shape. The axis starts
  // at the earliest iteration drawn rather than at zero: iterations cut away
  // are not on the chart, and leaving room for them would waste the width.
  const earliest = Math.min(...tracks.map(t => t.points[0].t));
  const longest = Math.max(...tracks.map(t => t.points[t.points.length - 1].t), 1);
  const mapT = t => (logX ? Metrics.applyLog(t, false) : t);
  const loT = mapT(earliest), hiT = Math.max(mapT(longest), loT + 1e-9);

  // A lone line's scale takes in its constant lines, with a little room
  // beyond: a threshold is most worth seeing when the data stays clear of it.
  // Then it is widened to the round numbers the axis labels, so the top and
  // bottom lines are labelled values rather than wherever the data stopped.
  const single = tracks.length === 1 ? tracks[0] : null;
  const yGuides = guides.filter(g => g.axis === 'y');
  if (single) {
    for (const guide of yGuides) {
      const v = mapY(guide.at);
      if (v > single.hi) single.hi = v + 0.05 * (v - single.lo);
      if (v < single.lo) single.lo = v - 0.05 * (single.hi - v);
    }
    if (single.hi - single.lo < 1e-12) { single.lo -= 0.5; single.hi += 0.5; }
    const ticks = _axisTicks(single.lo, single.hi, _tickCounts(w, h).down);
    if (ticks.length > 1) {
      single.lo = Math.min(single.lo, ticks[0]);
      single.hi = Math.max(single.hi, ticks[ticks.length - 1]);
    }
  }
  _axes(ctx, w, h, {
    x: { lo: loT, hi: hiT,
         format: v => formatNumber(Math.round(logX ? Metrics.undoLog(v, false) : v)) },
    y: single ? { lo: single.lo, hi: single.hi, format: yFormat }
              : { lo: 0, hi: 100, format: v => `${Math.round(v)}%` },
    grid: chrome ? chrome.grid !== false : true,
    guides: guides.filter(g => g.axis === 'x'), pad
  });

  for (const track of tracks) {
    if (track.hi === track.lo) track.hi = track.lo + 1;
    const span = track.stretch ? (mapT(track.points[track.points.length - 1].t) - mapT(track.points[0].t)) || 1
                               : (hiT - loT);
    const base = track.stretch ? mapT(track.points[0].t) : loT;
    const xAt = t => ((mapT(t) - base) / span) * w;
    const yAt = v => (1 - (v - track.lo) / (track.hi - track.lo)) * h;

    // Each line keeps its own vertical scale: these are different quantities
    // in different units, and one shared axis flattens whichever has the
    // smaller spread. A constant line is given in real units, so it is mapped
    // the same way the values were before it is compared against them.
    for (const guide of yGuides) {
      const value = mapY(guide.at);
      if (value < track.lo || value > track.hi) continue;
      ctx.save();
      ctx.setLineDash([5, 4]);
      ctx.strokeStyle = track.colour;
      ctx.globalAlpha = 0.5;
      const at = Math.round(yAt(value)) + 0.5;
      ctx.beginPath(); ctx.moveTo(0, at); ctx.lineTo(w, at); ctx.stroke();
      // A thesis names its threshold; a line added by hand has no name.
      if (guide.label) {
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = track.colour;
        ctx.font = Ink.font(11);
        ctx.fillText(guide.label, 6, at - 5);
      }
      ctx.restore();
    }

    ctx.strokeStyle = track.colour;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    track.points.forEach((p, k) => {
      const x = xAt(p.t), y = yAt(track.mapped[k]);
      if (k === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    });
    ctx.stroke();
  }

  if (chrome) {
    chrome.legend = tracks.filter(t => t.legend).map(t => ({ colour: t.colour, label: t.legend }));
    _chrome(ctx, pad, outer, chrome);
  }
}
