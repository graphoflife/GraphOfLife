/*
 * The front page's entrance: the title decoded out of the machine's digits, a
 * wall of them half lost in the dark at the edges, and the numbers of the
 * world turning behind it all.
 *
 * Each letter of the title arrives as hexadecimal, flickering through digits
 * until its turn comes, and lands as itself with a flash of phosphor. The
 * machine's alphabet first, then the word — which is the order the site asks
 * to be read in.
 *
 * Every moving part is advanced from Home.tick, which App calls only while the
 * front page is on screen and the browser is drawing it. So nothing here runs
 * behind another tab or in a hidden window: there are no timers of its own to
 * forget. CSS does the rest of the arrival (the world fading up, the numbers),
 * and the stylesheet holds all of that still for a reader who has asked for
 * less motion — as this does for the parts it moves.
 *
 * None of it touches the graph. The wall is its own canvas (glyphwall.js),
 * drawn once per size and then a few cells at a time, so the backdrop's
 * frame budget is the backdrop's alone.
 */
const Entrance = {
  /** The machine's alphabet. */
  DIGITS: '0123456789ABCDEF',

  /**
   * The title's schedule, in seconds: when the first letter starts, how far
   * apart the letters start, how long each flickers before it lands, and how
   * often a letter still in disguise changes its digit.
   */
  DECODE: { begin: 0.25, stagger: 0.085, hold: 0.85, every: 0.055 },

  /** When the subtitle starts typing, and the seconds per character. */
  TYPE: { begin: 1.65, every: 0.032 },

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

    this.buildTitle(document.getElementById('homeTitle'));
    this.buildSubtitle(document.getElementById('homeSubtitle'));
    this.wall = new GlyphWall(document.getElementById('homeGlyphs'), stage);
    this.readout = document.getElementById('homeReadout');
  },

  // ---- the title ------------------------------------------------------------

  /**
   * Each letter in a box of its own width, with a cover over it that shows
   * the digit it is pretending to be. The letter is there all along and only
   * made invisible, so the title takes its final width from the start and
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
        letter.append(cover);
        box.append(letter);
        this.letters.push({ el: letter, cover, at: 0, hidden: false, clock: 0 });
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

  /**
   * Hide a letter behind a digit, or let it land. A letter that lands is
   * marked for a moment, which is what its flash of phosphor hangs on.
   */
  disguise(letter, hidden) {
    if (letter.hidden !== hidden) {
      letter.hidden = hidden;
      letter.el.classList.toggle('is-hidden', hidden);
      letter.el.classList.toggle('is-landed', !hidden && !this.calm);
    }
    if (hidden) letter.cover.textContent = this.DIGITS[Math.floor(Math.random() * this.DIGITS.length)];
  },

  /** Everything where it ends: every letter itself, the subtitle typed, the caret gone. */
  settle() {
    for (const letter of this.letters || []) this.disguise(letter, false);
    this.type(this.subtitleText?.length || 0);
    this.caret?.classList.add('is-done');
    this.decoding = false;
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
    this.wall?.resize();
    if (this.calm) {
      this.settle();
      return;
    }
    this.decoding = true;
    (this.letters || []).forEach((letter, i) => {
      letter.at = this.DECODE.begin + i * this.DECODE.stagger + Math.random() * 0.06;
      letter.clock = 0;
      letter.el.classList.remove('is-landed');
      this.disguise(letter, true);
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
    if (this.active) this.wall?.draw();
  },

  /** The theme changed: the wall is drawn again in its inks. */
  repaint() {
    if (this.active) this.wall?.draw();
  },

  /** One animation frame, handed on by Home while the front page is shown. */
  tick(dt) {
    if (!this.active) return;
    this.time += dt;
    if (this.calm) return;
    if (this.decoding) this.tickTitle(dt);
    this.wall?.tick(dt);
  },

  tickTitle(dt) {
    const { hold, every } = this.DECODE;
    let pending = false;
    for (const letter of this.letters || []) {
      if (!letter.hidden) continue;
      const hidden = this.time - letter.at < hold;
      letter.clock += dt;
      if (!hidden || letter.clock >= every) {
        letter.clock = 0;
        this.disguise(letter, hidden);
      }
      if (hidden) pending = true;
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
