/*
 * The colour themes, and the switch in the top bar that chooses one.
 *
 * A theme is a set of tokens in style.css, chosen by the name on <html
 * data-theme="…">; Obsidian, the stylesheet's base, has no name there. The
 * choice is kept in this browser (localStorage), and a two-line script in
 * index.html's <head> puts it back before anything is painted, so a white
 * page never flashes dark on the way in. Until a browser chooses, that script
 * gives it Phosphor.
 *
 * Changing theme changes the stylesheet's colours at once, but not what was
 * painted on a canvas: those read their inks through Ink, once. So the change
 * is announced — Ink forgets what it read, and a `themechange` event goes out
 * to whatever draws (the Viewer, the front page, the charts) to draw again.
 */
const Theme = {
  STORE: 'gol.theme',

  /** The themes, in the order the switch offers them: [name, what it is called]. */
  THEMES: [['obsidian', 'Obsidian'], ['phosphor', 'Phosphor'],
           ['midnight', 'Midnight'], ['paper', 'Paper']],

  current: 'phosphor',

  init() {
    const set = document.documentElement.dataset.theme;
    this.current = this.THEMES.some(([name]) => name === set) ? set : 'obsidian';

    this.el = document.getElementById('themeSwitch');
    if (!this.el) return;
    this.el.replaceChildren(...this.THEMES.map(([name, label]) => {
      const button = document.createElement('button');
      button.type = 'button';
      button.className = `theme-swatch theme-swatch-${name}`;
      button.dataset.theme = name;
      button.title = `${label} theme`;
      button.setAttribute('aria-label', `${label} theme`);
      button.addEventListener('click', () => this.set(name));
      return button;
    }));
    this.mark();
  },

  /** Show which swatch is the theme in use. */
  mark() {
    for (const button of this.el ? this.el.querySelectorAll('.theme-swatch') : []) {
      button.setAttribute('aria-pressed', String(button.dataset.theme === this.current));
    }
  },

  set(name) {
    if (name === this.current || !this.THEMES.some(([known]) => known === name)) return;
    this.current = name;
    if (name === 'obsidian') delete document.documentElement.dataset.theme;
    else document.documentElement.dataset.theme = name;
    try { localStorage.setItem(this.STORE, name); } catch (err) { /* a private window keeps it for the visit */ }
    this.mark();
    Ink.reset();
    window.dispatchEvent(new CustomEvent('themechange', { detail: { theme: name } }));
  }
};
