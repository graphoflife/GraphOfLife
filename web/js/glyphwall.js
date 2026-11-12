/*
 * A wall of the machine's glyphs, half lost in the dark: hexadecimal digits
 * in a loose grid, each at an opacity of its own, now and then one becoming
 * another and flaring before it cools back into the wall.
 *
 * Behind the front page's title it is emptier towards the middle, where the
 * words stand, and fuller towards the edges; behind the Explanation's picture
 * it is even and fainter still. Either way it is a texture, not something to
 * read, so it costs as little as a texture should: drawn whole once for each
 * size, and after that only the few cells that are changing, at most fifteen
 * times a second. Nothing moves unless its owner calls tick(), which the
 * owner does only while its page is on screen.
 */

/**
 * A small seeded generator (mulberry32). A wall is drawn again whenever
 * anything asks the page to measure itself — the web font arriving does —
 * and drawn from Math.random it would reshuffle in front of the reader.
 */
function seededRandom(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Words decoded into place, as the front page's title is: each character
 * arrives as a hexadecimal digit, flickers, and lands as itself, left to
 * right. For the head of a page and the Explanation's step titles. Every
 * character keeps its own width while it is in disguise, so the line never
 * reflows as it resolves, and a screen reader is given the words whole.
 *
 * Its own short loop, of a second or so, which stops as soon as the words
 * have landed or are no longer on screen; nothing at all for a reader who
 * asked for less motion.
 */
const Decoder = {
  calm() {
    return !!(window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
  },

  decode(el, { begin = 0.12, stagger = 0.022 } = {}) {
    if (!el) return;
    // The words as written, kept the first time, so decoding again starts
    // from them rather than from the letters and covers of the last time.
    const text = el.dataset.words || el.textContent.trim();
    el.dataset.words = text;
    if (this.calm()) { el.textContent = text; return; }
    el.textContent = '';
    const said = Object.assign(document.createElement('span'), { className: 'sr-only', textContent: text });
    const shown = document.createElement('span');
    shown.setAttribute('aria-hidden', 'true');
    const letters = [];
    for (const word of text.split(/(\s+)/)) {
      if (!word) continue;
      if (/^\s+$/.test(word)) { shown.append(word); continue; }
      const box = Object.assign(document.createElement('span'), { className: 'decoded-word' });
      for (const ch of word) {
        const letter = Object.assign(document.createElement('span'), { className: 'glyph is-hidden', textContent: ch });
        const cover = Object.assign(document.createElement('i'), { className: 'glyph-cover' });
        letter.append(cover);
        box.append(letter);
        letters.push({ el: letter, cover });
      }
      shown.append(box);
    }
    el.append(said, shown);

    const glyphs = GlyphWall.GLYPHS;
    let last = null, time = 0, clock = 1;
    const step = now => {
      const dt = last === null ? 0 : Math.min(0.05, (now - last) / 1000);
      last = now;
      time += dt;
      clock += dt;
      const flicker = clock >= 0.05;
      if (flicker) clock = 0;
      // Off the page or out of view (its tab left): land everything at once.
      const gone = !el.isConnected || el.offsetParent === null;
      let pending = false;
      letters.forEach((letter, i) => {
        if (!letter.el.classList.contains('is-hidden')) return;
        if (gone || time >= begin + i * stagger) {
          letter.el.classList.remove('is-hidden');
          if (!gone) letter.el.classList.add('is-landed');
          return;
        }
        pending = true;
        if (flicker) letter.cover.textContent = glyphs[Math.floor(Math.random() * glyphs.length)];
      });
      if (pending) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }
};

class GlyphWall {
  /**
   * `canvas` is drawn on, sized to `host`. The options say where the wall is
   * thin: `hole` is how far from the middle (as a share of the half-size) it
   * stays empty, `span` how far beyond that it takes to fill, and `density`
   * the share of cells that hold a glyph where it is full.
   */
  constructor(canvas, host, options = {}) {
    this.canvas = canvas;
    this.host = host;
    this.options = Object.assign({
      cell: [22, 28], smallCell: [17, 23], seed: 1729,
      hole: 0.5, span: 0.65, density: 0.5, centreY: 0.46,
      faint: 0.05, range: 0.09, lit: 0.025,
      changeEvery: 0.11, cool: 1.6, rate: 15
    }, options);
    this.cells = [];
    this.hot = new Set();
    this.size = '';
  }

  /** Hexadecimal, the machine's own alphabet. */
  static get GLYPHS() { return '0123456789ABCDEF'; }

  someGlyph(pick) {
    const glyphs = GlyphWall.GLYPHS;
    return glyphs[Math.floor(pick * glyphs.length) % glyphs.length];
  }

  /** The cells, laid out for a wall of this size. */
  layout(width, height) {
    const o = this.options;
    const [cw, ch] = width < 640 ? o.smallCell : o.cell;
    const rand = seededRandom(o.seed);
    const cells = [];
    const cols = Math.ceil(width / cw), rows = Math.ceil(height / ch);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = (c + 0.5) * cw, y = (r + 0.5) * ch;
        const dx = (x - width / 2) / (width / 2), dy = (y - height * o.centreY) / (height / 2);
        const reach = Math.hypot(dx, dy * 1.12);
        const chance = Math.pow(Math.min(1, Math.max(0, (reach - o.hole) / o.span)), 1.4) * o.density;
        const roll = rand(), pick = rand(), shade = rand();
        if (roll > chance) continue;
        const lit = shade > 1 - o.lit;
        const base = lit ? o.faint * 5 + o.range * rand() : o.faint + o.range * shade;
        cells.push({ x, y, w: cw, h: ch, base, glow: 0, glyph: this.someGlyph(pick) });
      }
    }
    return cells;
  }

  /** The whole wall, from scratch: on arrival, and whenever its size or theme changes. */
  draw() {
    const canvas = this.canvas, host = this.host;
    if (!canvas || !host) return;
    const width = host.clientWidth, height = host.clientHeight;
    if (!(width > 0 && height > 0)) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Set once: parsing a font for every glyph is the one thing here that
    // would add up.
    ctx.font = Ink.font(11);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    this.ctx = ctx;
    this.ink = Ink.of('accent');
    this.cells = this.layout(width, height);
    this.hot = new Set();
    ctx.clearRect(0, 0, width, height);
    for (const cell of this.cells) this.drawCell(cell, false);
    this.size = `${width}x${height}`;
  }

  /** Drawn again only if the host has changed size since. */
  resize() {
    if (this.host && this.size !== `${this.host.clientWidth}x${this.host.clientHeight}`) this.draw();
  }

  /** One cell, at its opacity now: its resting shade plus whatever glow it still has. */
  drawCell(cell, clear = true) {
    const ctx = this.ctx;
    if (clear) ctx.clearRect(cell.x - cell.w / 2, cell.y - cell.h / 2, cell.w, cell.h);
    ctx.fillStyle = Ink.alpha(this.ink, Math.min(0.9, cell.base + cell.glow).toFixed(3));
    ctx.fillText(cell.glyph, cell.x, cell.y);
  }

  /**
   * The wall, alive: now and then one glyph becomes another and flares, then
   * cools. Only the cells that are cooling are drawn.
   */
  tick(dt) {
    if (!this.cells.length || !this.ctx) return;
    const o = this.options;
    this._clock = (this._clock || 0) + dt;
    this._change = (this._change || 0) + dt;
    while (this._change >= o.changeEvery) {
      this._change -= o.changeEvery;
      const cell = this.cells[Math.floor(Math.random() * this.cells.length)];
      cell.glyph = this.someGlyph(Math.random());
      cell.glow = 0.42;
      this.hot.add(cell);
    }
    if (this._clock < 1 / o.rate) return;
    const step = this._clock;
    this._clock = 0;
    for (const cell of this.hot) {
      cell.glow = Math.max(0, cell.glow - 0.42 * step / o.cool);
      this.drawCell(cell);
      if (cell.glow === 0) this.hot.delete(cell);
    }
  }
}
