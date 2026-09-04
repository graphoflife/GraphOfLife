/*
 * The small emblems the interface draws: an eye, a heart, a skull, crossed
 * swords, a flag, a helix.
 *
 * One light grey on nothing — no second colour, no fill behind the strokes —
 * so an emblem reads as a mark on whatever it sits on rather than as an
 * illustration competing with it.
 *
 * Kept here rather than beside the one screen that first wanted them, because
 * the same eye now stands for "look at this" in two places and they must not
 * drift into being two different eyes. Anything that needs one asks for it by
 * name and gets a colour and an opacity of its choosing.
 */
const Emblems = {
  DEFAULT_INK: '#c8cfd9',

  SHAPES: {
    eye: `<path d="M2.2 12C5.6 6.8 8.7 4.6 12 4.6s6.4 2.2 9.8 7.4c-3.4 5.2-6.5 7.4-9.8 7.4S5.6 17.2 2.2 12z"/>
          <circle cx="12" cy="12" r="3.5" FILL/>`,
    heart: `<path d="M12 21.2S2.6 14.9 2.6 8.9A5.4 5.4 0 0 1 12 5.6a5.4 5.4 0 0 1 9.4 3.3c0 6-9.4 12.3-9.4 12.3z" FILL/>`,
    skull: `<path fill-rule="evenodd" d="M12 1.8c-5 0-8.7 3.6-8.7 8.2 0 2.6 1.2 4.5 2.7 5.6v2.2c0 .9.7 1.6 1.6 1.6h.7v2.8h2.1v-2.8h3.2v2.8h2.1v-2.8h.7c.9 0 1.6-.7 1.6-1.6v-2.2c1.5-1.1 2.7-3 2.7-5.6 0-4.6-3.7-8.2-8.7-8.2zM8.6 8.1a2.1 2.1 0 1 0 0 4.2 2.1 2.1 0 0 0 0-4.2zm6.8 0a2.1 2.1 0 1 0 0 4.2 2.1 2.1 0 0 0 0-4.2zM12 13.4l1.3 2.6h-2.6z" FILL/>`,
    swords: `<path d="M4 3.5 18.5 19.5M20 3.5 5.5 19.5M16.9 14.5 13.7 17.5M7.1 14.5 10.3 17.5M15.3 16 18.5 19.5M8.7 16 5.5 19.5"/>`,
    flag: `<path d="M5.5 2.6V21.4"/><path d="M5.5 3.9c4.1-2.1 7.2 2 11.3 0v8.2c-4.1 2.1-7.2-2-11.3 0z" FILL/>`,
    dna: `<path d="M12.00 2.60C13.30 3.12 14.60 3.64 15.52 4.17C16.45 4.69 16.99 5.21 17.00 5.73C17.01 6.26 16.49 6.78 15.58 7.30C14.67 7.82 13.38 8.34 12.08 8.87C10.78 9.39 9.48 9.91 8.54 10.43C7.60 10.96 7.03 11.48 7.00 12.00C6.97 12.52 7.47 13.04 8.36 13.57C9.26 14.09 10.53 14.61 11.83 15.13C13.14 15.66 14.45 16.18 15.40 16.70C16.36 17.22 16.94 17.74 17.00 18.27C17.05 18.79 16.57 19.31 15.69 19.83C14.82 20.36 13.55 20.88 12.25 21.40"/><path d="M12.00 2.60C10.70 3.12 9.40 3.64 8.48 4.17C7.55 4.69 7.01 5.21 7.00 5.73C6.99 6.26 7.51 6.78 8.42 7.30C9.33 7.82 10.62 8.34 11.92 8.87C13.22 9.39 14.52 9.91 15.46 10.43C16.40 10.96 16.97 11.48 17.00 12.00C17.03 12.52 16.53 13.04 15.64 13.57C14.74 14.09 13.47 14.61 12.17 15.13C10.86 15.66 9.55 16.18 8.60 16.70C7.64 17.22 7.06 17.74 7.00 18.27C6.95 18.79 7.43 19.31 8.31 19.83C9.18 20.36 10.45 20.88 11.75 21.40"/><path d="M8.90 3.94H15.10M7.47 6.63H16.53M7.00 12.00H17.00M7.58 17.37H16.42M8.70 20.06H15.30"/>`
  },

  /**
   * One emblem as a url(...) a stylesheet can use.
   *
   * `FILL` in a shape marks the parts that are solid rather than stroked; it
   * is substituted here so a shape reads as a shape rather than as a pile of
   * repeated attributes.
   */
  url(name, { ink = this.DEFAULT_INK, opacity = 1, width = 1.6 } = {}) {
    const body = this.SHAPES[name];
    if (!body) return 'none';
    return `url("data:image/svg+xml,${encodeURIComponent(this.svg(name, { ink, opacity, width }))}")`;
  },

  /** The same thing as markup, for somewhere that wants to inline it. */
  svg(name, { ink = this.DEFAULT_INK, opacity = 1, width = 1.6 } = {}) {
    const body = this.SHAPES[name];
    if (!body) return '';
    return `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" `
      + `fill="none" stroke="${ink}" stroke-width="${width}" `
      + `stroke-linecap="round" stroke-linejoin="round" opacity="${opacity}">`
      + body.replace(/FILL/g, `fill="${ink}" stroke="none"`)
      + `</svg>`;
  }
};
