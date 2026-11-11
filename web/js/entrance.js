/*
 * The front page's entrance: the title decoded out of the machine's digits and
 * the runes, a ring of runes carved around the mark, an inscription half lost
 * in the dark at the edges, and the numbers of the world turning behind it.
 *
 * Each letter of the title passes through three alphabets on its way in: the
 * hexadecimal of the machine, then a rune, then itself. Code, then the old
 * script, then the word — which is the order the site asks to be read in.
 *
 * Every moving part is advanced from Home.tick, which App calls only while the
 * front page is on screen and the browser is drawing it. So nothing here runs
 * behind another tab or in a hidden window: there are no timers of its own to
 * forget. CSS does the rest of the arrival (the ring drawing itself, the world
 * fading up, the doors), and the stylesheet holds all of that still for a
 * reader who has asked for less motion — as this does for the parts it moves.
 *
 * None of it touches the graph. The inscription is its own canvas, drawn once
 * per size and then a few cells at a time, so the backdrop's frame budget is
 * the backdrop's alone.
 */
const Entrance = {
  /** The runes, by their id in the page's sprite (index.html), in futhark order. */
  RUNES: ['fehu', 'uruz', 'thurisaz', 'ansuz', 'raidho', 'kaunan', 'gebo', 'wunjo', 'naudiz',
          'isa', 'jera', 'eihwaz', 'perthro', 'berkano', 'ehwaz', 'mannaz', 'laguz', 'ingwaz', 'dagaz'],

  /** The machine's alphabet. */
  DIGITS: '0123456789ABCDEF',

  /**
   * The title's schedule, in seconds: when the first letter starts, how far
   * apart the letters start, how long each spends as digits and then as a
   * rune, and how often a letter still in disguise changes its glyph.
   */
  DECODE: { begin: 0.25, stagger: 0.085, digits: 0.42, runes: 0.5, every: 0.055 },

  /** When the subtitle starts typing, and the seconds per character. */
  TYPE: { begin: 1.65, every: 0.032 },

  /**
   * The inscription: the cell a glyph sits in, in CSS pixels, how often one
   * of them changes, and how long a changed glyph takes to fade back into the
   * wall. Redrawn at most this many times a second — it is a texture, and
   * fifteen is plenty for one.
   */
  FIELD: { cell: [22, 28], smallCell: [17, 23], changeEvery: 0.11, cool: 1.6, rate: 15 },

  stage: null,
  active: false,
  calm: false,
  time: 0,

  init(stage) {
    this.stage = stage;
    if (!stage) return;

    // A reader who has asked for less motion gets the page as it ends: no
    // decoding, no typing, nothing turning. Watched, so changing the setting
    // takes effect without a reload.
    const query = window.matchMedia ? window.matchMedia('(prefers-reduced-motion: reduce)') : null;
    this.calm = !!(query && query.matches);
    query?.addEventListener?.('change', event => {
      this.calm = event.matches;
      if (this.calm) this.settle();
    });

    this.runePaths = this.readRunes();
    this.buildTitle(document.getElementById('homeTitle'));
    this.buildSubtitle(document.getElementById('homeSubtitle'));
    this.buildRing(document.getElementById('homeRing'));
    this.field = document.getElementById('homeGlyphs');
    this.readout = document.getElementById('homeReadout');
  },

  /** The runes' strokes, read from the sprite, for the canvas to draw. */
  readRunes() {
    const paths = {};
    for (const name of this.RUNES) {
      const d = document.querySelector(`#rune-${name} path`)?.getAttribute('d');
      if (d && window.Path2D) paths[name] = new Path2D(d);
    }
    return paths;
  },

  // ---- the title ------------------------------------------------------------

  /**
   * Each letter in a box of its own width, with a cover over it that shows
   * what the letter is pretending to be. The letter is there all along and
   * only made invisible, so the title takes its final width from the start and
   * nothing shifts as it resolves. A screen reader is given the words whole.
   */
  buildTitle(el) {
    if (!el) return;
    const words = el.textContent.trim().split(/\s+/);
    el.textContent = '';
    const said = document.createElement('span');
    said.className = 'sr-only';
    said.textContent = words.join(' ');
    const shown = document.createElement('span');
    shown.setAttribute('aria-hidden', 'true');
    this.letters = [];
    words.forEach((word, w) => {
      const box = document.createElement('span');
      box.className = 'home-word';
      for (const ch of word) {
        const letter = document.createElement('span');
        letter.className = 'glyph';
        letter.textContent = ch;
        const cover = document.createElement('i');
        cover.className = 'glyph-cover';
        cover.innerHTML = '<b></b><svg class="rune" viewBox="0 0 10 16"><use/></svg>';
        letter.append(cover);
        box.append(letter);
        this.letters.push({ el: letter, cover, digit: cover.firstChild,
                            use: cover.querySelector('use'), at: 0, mode: '' });
      }
      shown.append(box);
      if (w < words.length - 1) shown.append(' ');
    });
    el.append(said, shown);
  },

  /**
   * The subtitle, typed: the words whole for a screen reader, and for the eye
   * what has been typed, the cursor, and what is still to come — there all
   * along, unseen, so the line is its full width from the start and stays
   * centred as it fills.
   */
  buildSubtitle(el) {
    if (!el) return;
    this.subtitleText = el.textContent.trim();
    el.textContent = '';
    const said = document.createElement('span');
    said.className = 'sr-only';
    said.textContent = this.subtitleText;
    const part = (cls, text = '') => {
      const span = document.createElement('span');
      if (cls) span.className = cls;
      span.setAttribute('aria-hidden', 'true');
      span.textContent = text;
      return span;
    };
    this.typed = part('', this.subtitleText);
    this.caret = part('home-caret is-done');
    this.untyped = part('home-untyped');
    el.append(said, this.typed, this.caret, this.untyped);
  },

  /** The first `n` characters of the subtitle typed, the rest still to come. */
  type(n) {
    if (!this.typed) return;
    this.typed.textContent = this.subtitleText.slice(0, n);
    this.untyped.textContent = this.subtitleText.slice(n);
  },

  /** Disguise a letter as the machine's digits or as a rune. */
  disguise(letter, mode) {
    if (letter.mode !== mode) {
      letter.mode = mode;
      letter.cover.dataset.mode = mode;
      letter.el.classList.toggle('is-hidden', mode !== '');
    }
    if (mode === 'digit') {
      letter.digit.textContent = this.DIGITS[Math.floor(Math.random() * this.DIGITS.length)];
    } else if (mode === 'rune') {
      letter.use.setAttribute('href', `#rune-${this.RUNES[Math.floor(Math.random() * this.RUNES.length)]}`);
    }
  },

  /** Everything where it ends: every letter itself, the subtitle typed, the caret gone. */
  settle() {
    for (const letter of this.letters || []) this.disguise(letter, '');
    this.type(this.subtitleText?.length || 0);
    this.caret?.classList.add('is-done');
    this.decoding = false;
  },

  // ---- the ring -------------------------------------------------------------

  /**
   * The runes around the mark: the futhark once, each followed by a point, the
   * way inscriptions part their words, between two circles. Each knows its
   * place in the ring (--i), so the stylesheet can light them in order.
   */
  buildRing(svg) {
    if (!svg) return;
    const NS = 'http://www.w3.org/2000/svg';
    const make = (tag, attrs) => {
      const el = document.createElementNS(NS, tag);
      for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
      return el;
    };
    svg.append(make('circle', { class: 'home-ring-line', r: 97, pathLength: 1 }),
               make('circle', { class: 'home-ring-line', r: 80, pathLength: 1 }));
    const slots = this.RUNES.length * 2;
    const band = make('g', { class: 'home-ring-band' });
    this.RUNES.forEach((name, i) => {
      const angle = (i * 2) * 360 / slots;
      const rune = make('use', { href: `#rune-${name}`, x: -5, y: -8, width: 10, height: 16,
                                 transform: `rotate(${angle.toFixed(2)}) translate(0 -88.5) scale(0.74)` });
      rune.style.setProperty('--i', i);
      band.append(rune);
      const dot = make('circle', { class: 'home-ring-point', r: 0.95,
                                   transform: `rotate(${(angle + 360 / slots).toFixed(2)}) translate(0 -88.5)` });
      dot.style.setProperty('--i', i);
      band.append(dot);
    });
    svg.append(band);
    // Four notches on the outer circle, at the quarters, like the marks on a dial.
    for (let q = 0; q < 4; q++) {
      svg.append(make('rect', { class: 'home-ring-notch', x: -2, y: -2, width: 4, height: 4,
                                transform: `rotate(${q * 90}) translate(0 -97) rotate(45)` }));
    }
  },

  // ---- the inscription ------------------------------------------------------

  /**
   * A small seeded generator. The wall is drawn again whenever anything asks
   * the page to measure itself — the web font arriving does — and drawn from
   * Math.random it would reshuffle in front of the reader each time.
   */
  random(seed) {
    let a = seed >>> 0;
    return () => {
      a = (a + 0x6D2B79F5) >>> 0;
      let t = a;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  },

  /**
   * Lay out the wall: a grid of cells, emptier towards the middle where the
   * title stands and fuller towards the edges, each holding a rune or a digit
   * at an opacity of its own. The odd one is lit.
   */
  layoutField(width, height) {
    const [cw, ch] = width < 640 ? this.FIELD.smallCell : this.FIELD.cell;
    const rand = this.random(1729);
    const cells = [];
    const cols = Math.ceil(width / cw), rows = Math.ceil(height / ch);
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++) {
        const x = (c + 0.5) * cw, y = (r + 0.5) * ch;
        const dx = (x - width / 2) / (width / 2), dy = (y - height * 0.46) / (height / 2);
        const reach = Math.hypot(dx, dy * 1.12);
        const chance = Math.pow(Math.min(1, Math.max(0, (reach - 0.5) / 0.65)), 1.4) * 0.5;
        const roll = rand(), pick = rand(), shade = rand();
        if (roll > chance) continue;
        const lit = shade > 0.975;
        const base = lit ? 0.26 + 0.12 * rand() : 0.05 + 0.09 * shade;
        cells.push({ x, y, w: cw, h: ch, base, glow: 0, ...this.someGlyph(pick) });
      }
    }
    return cells;
  },

  /** A rune a little under half the time, a digit otherwise, both chosen by `pick` in [0, 1). */
  someGlyph(pick) {
    const RUNE = 0.45;
    return pick < RUNE
      ? { kind: 'rune', rune: this.RUNES[Math.floor(pick / RUNE * this.RUNES.length)] }
      : { kind: 'digit', digit: this.DIGITS[Math.floor((pick - RUNE) / (1 - RUNE) * this.DIGITS.length)] };
  },

  /** The whole wall, from scratch: on arrival and whenever the stage changes size. */
  drawField() {
    const canvas = this.field;
    if (!canvas || !this.stage) return;
    const width = this.stage.clientWidth, height = this.stage.clientHeight;
    if (!(width > 0 && height > 0)) return;
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    canvas.width = Math.round(width * dpr);
    canvas.height = Math.round(height * dpr);
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    // Set once: a rune's save and restore keep them, and parsing a font for
    // every digit is the one thing here that would add up.
    ctx.font = Ink.font(11);
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    this.ctx = ctx;
    this.inks = { rune: Ink.of('gold'), digit: Ink.of('accent') };
    this.cells = this.layoutField(width, height);
    this.hot = new Set();
    ctx.clearRect(0, 0, width, height);
    for (const cell of this.cells) this.drawCell(cell, false);
    this.fieldSize = `${width}x${height}`;
  },

  /** One cell, at its opacity now: its resting shade plus whatever glow it still has. */
  drawCell(cell, clear = true) {
    const ctx = this.ctx;
    if (clear) ctx.clearRect(cell.x - cell.w / 2, cell.y - cell.h / 2, cell.w, cell.h);
    const alpha = Math.min(0.9, cell.base + cell.glow).toFixed(3);
    if (cell.kind === 'rune') {
      const path = this.runePaths[cell.rune];
      if (!path) return;
      ctx.save();
      ctx.translate(cell.x - 3.6, cell.y - 5.8);
      ctx.scale(0.72, 0.72);
      ctx.lineWidth = 1.35;
      ctx.strokeStyle = Ink.alpha(this.inks.rune, alpha);
      ctx.stroke(path);
      ctx.restore();
    } else {
      ctx.fillStyle = Ink.alpha(this.inks.digit, alpha);
      ctx.fillText(cell.digit, cell.x, cell.y);
    }
  },

  /**
   * The wall, alive: now and then one glyph becomes another and flares, then
   * cools back into the stone. Only the cells that are cooling are drawn.
   */
  tickField(dt) {
    if (!this.cells || !this.cells.length) return;
    this._fieldClock = (this._fieldClock || 0) + dt;
    this._changeClock = (this._changeClock || 0) + dt;
    while (this._changeClock >= this.FIELD.changeEvery) {
      this._changeClock -= this.FIELD.changeEvery;
      const cell = this.cells[Math.floor(Math.random() * this.cells.length)];
      Object.assign(cell, this.someGlyph(Math.random()), { glow: 0.42 });
      this.hot.add(cell);
    }
    if (this._fieldClock < 1 / this.FIELD.rate) return;
    const step = this._fieldClock;
    this._fieldClock = 0;
    for (const cell of this.hot) {
      cell.glow = Math.max(0, cell.glow - 0.42 * step / this.FIELD.cool);
      this.drawCell(cell);
      if (cell.glow === 0) this.hot.delete(cell);
    }
  },

  // ---- the numbers ----------------------------------------------------------

  /** The world on screen, in three numbers; one that changed flickers as it lands. */
  report(frame) {
    if (!this.readout || !frame) return;
    let tokens = 0;
    for (const t of frame.tokens) tokens += t;
    const counts = { agents: frame.ids.length, links: frame.edges.length, tokens };
    for (const [key, value] of Object.entries(counts)) {
      const el = this.readout.querySelector(`[data-count="${key}"]`);
      if (!el) continue;
      const text = value.toLocaleString('en-US');
      if (el.textContent === text) continue;
      el.textContent = text;
      if (this.calm) continue;
      el.classList.remove('is-new');
      void el.offsetWidth;
      el.classList.add('is-new');
    }
  },

  // ---- opened, left, and every frame between --------------------------------

  /** The page is opened: the title is in disguise again and the decoding starts over. */
  start() {
    if (!this.stage) return;
    this.active = true;
    this.time = 0;
    if (this.fieldSize !== `${this.stage.clientWidth}x${this.stage.clientHeight}`) this.drawField();
    if (this.calm) {
      this.settle();
      return;
    }
    this.decoding = true;
    (this.letters || []).forEach((letter, i) => {
      letter.at = this.DECODE.begin + i * this.DECODE.stagger + Math.random() * 0.06;
      letter.clock = 0;
      this.disguise(letter, 'digit');
    });
    this.type(0);
    this.caret?.classList.remove('is-done');
  },

  /** The page is left. Nothing keeps running; the next visit starts from the top. */
  stop() {
    this.active = false;
    this.decoding = false;
  },

  /** The stage changed size: the wall is laid out again to fit it. */
  resize() {
    if (this.active) this.drawField();
  },

  /** One animation frame, handed on by Home while the front page is shown. */
  tick(dt) {
    if (!this.active) return;
    this.time += dt;
    if (this.calm) return;
    if (this.decoding) this.tickTitle(dt);
    this.tickField(dt);
  },

  tickTitle(dt) {
    const { digits, runes, every } = this.DECODE;
    let pending = false;
    for (const letter of this.letters || []) {
      if (letter.mode === '') continue;
      const age = this.time - letter.at;
      const mode = age < digits ? 'digit' : age < digits + runes ? 'rune' : '';
      letter.clock += dt;
      if (mode !== letter.mode || letter.clock >= every) {
        letter.clock = 0;
        this.disguise(letter, mode);
      }
      if (mode !== '') pending = true;
    }

    // The subtitle types itself once most of the title has landed.
    const length = this.subtitleText?.length || 0;
    const typed = Math.max(0, Math.floor((this.time - this.TYPE.begin) / this.TYPE.every));
    const shown = Math.min(length, typed);
    if (this.typed && this.typed.textContent.length !== shown) this.type(shown);
    const typing = shown < length;
    if (!pending && !typing) {
      this.decoding = false;
      this.caret?.classList.add('is-done');
    }
  }
};
