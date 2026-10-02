/*
 * The page's colours, for code that paints rather than styles.
 *
 * A canvas cannot say var(--muted); it wants a colour string. Written out as
 * hex, those strings were a second palette that nothing kept in step with the
 * first: when the panels were lightened, the stat popup's grid ended up darker
 * than the panel it sat on (1.05:1) and its labels at 2.62:1, and the contrast
 * audit could not see either, because text on a canvas is not text to it.
 *
 * So a chart asks for a colour by what it is for, and the stylesheet decides
 * what that is. The roles are few on purpose: every word a chart draws is one
 * of three greys, and every line of its frame one of three strengths.
 */
const Ink = {
  ROLES: {
    text: '--text',          // a title, a marker's outline: read first
    label: '--muted',        // tick values, legends, a line the reader added
    dim: '--muted-2',        // axis names, footnotes, an empty chart's message
    accent: '--accent',      // a lone series with no colour of its own
    grid: '--chart-grid',
    tick: '--chart-tick',
    axis: '--chart-axis'
  },

  /**
   * Colours a line gets when several share one chart, in order: the Diagrams
   * tab's and the book's figures alike. Taken from the page's own palette
   * rather than a colormap, because a colormap is a gradient and these have
   * to be told apart, not ordered.
   */
  LINES: ['#5ac8fa', '#ffd166', '#ff6b6b', '#7ee787', '#c792ea',
          '#f78c6c', '#89ddff', '#e5e5e5'],

  /** The colour of the `i`th line on a chart. */
  line(i) {
    return this.LINES[i % this.LINES.length];
  },

  _read: null,

  /**
   * The colour for a role, read from the stylesheet once.
   *
   * Loud rather than forgiving. An unknown role, or a token the stylesheet no
   * longer defines, would otherwise hand the canvas an empty string — which
   * it ignores, keeping whatever colour it had, so the mistake would show up
   * as a label quietly drawn in some other element's colour.
   */
  of(role) {
    if (!this._read) {
      const style = getComputedStyle(document.documentElement);
      this._read = {};
      for (const [name, token] of Object.entries(this.ROLES)) {
        const value = style.getPropertyValue(token).trim();
        if (!value) throw new Error(`the stylesheet does not define ${token}`);
        this._read[name] = value;
      }
    }
    if (!(role in this._read)) throw new Error(`no ink for "${role}"`);
    return this._read[role];
  }
};
