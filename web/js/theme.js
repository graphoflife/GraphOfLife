/*
 * The colour themes, and the switch in the top bar that chooses one.
 *
 * A theme is a set of tokens in style.css, chosen by the name on <html
 * data-theme="…">; Obsidian, the stylesheet's base, has no name there. The
 * choice is kept in this browser (localStorage), and a two-line script in
 * index.html's <head> puts it back before anything is painted, so a white
 * page never flashes dark on the way in. Until a browser chooses, that script
 * gives it Phosphor. The swatches are written in index.html, one for each
 * theme the switch offers, so the bar is whole from the first paint; this only
 * makes them work.
 *
 * Changing theme changes the stylesheet's colours at once, but not what was
 * painted on a canvas: those read their inks through Ink, once. So the change
 * is announced — Ink forgets what it read, and a `themechange` event goes out
 * to whatever draws (the Viewer, the front page, the charts) to draw again.
 */
const Theme = {
  STORE: 'gol.theme',

  /** The switch's swatches, one per theme, in the order it offers them. */
  swatches: [],

  current: 'phosphor',

  init() {
    this.swatches = [...document.querySelectorAll('#themeSwitch [data-theme]')];
    const set = document.documentElement.dataset.theme;
    this.current = this.offers(set) ? set : 'obsidian';
    for (const button of this.swatches) button.addEventListener('click', () => this.set(button.dataset.theme));
    this.mark();
  },

  /** Whether the switch offers a theme of this name. */
  offers(name) {
    return this.swatches.some(button => button.dataset.theme === name);
  },

  /** Show which swatch is the theme in use. */
  mark() {
    for (const button of this.swatches) {
      button.setAttribute('aria-pressed', String(button.dataset.theme === this.current));
    }
  },

  set(name) {
    if (name === this.current || !this.offers(name)) return;
    this.current = name;
    if (name === 'obsidian') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = name;
    try { localStorage.setItem(this.STORE, name); } catch (err) { /* a private window keeps it for the visit */ }
    this.mark();
    Ink.reset();
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: name } }));
  }
};
