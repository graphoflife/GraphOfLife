/* Tab switching, panel resizing, the animation loop, and start-up. */
const App = {
  view: 'home',

  /**
   * The views, by the name their tab carries. Each is told when it is shown
   * and when it is left (setActive), may be asked to pick up what hiding the
   * window cut short (resume), and, if it moves, is handed every animation
   * frame while it is on screen (tick). Opening, leaving and animating a view
   * were written out by name here, a branch at a time, and a view whose
   * branch was missing was how Research came back blank.
   */
  views: null,

  init() {
    // The wordmark carries data-view too, so it is a way back to the front
    // page rather than decoration.
    for (const tab of document.querySelectorAll('[data-view]')) {
      tab.addEventListener('click', () => this.showView(tab.dataset.view));
    }

    Jobs.attach(document.getElementById('jobBar'));

    // A hidden tab is not a tab anyone is reading. Work started in it kept
    // fetching and kept the layout worker spinning, which is most of what
    // "something in another window is still loading" was.
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) Jobs.cancelAll();
      else this.reloadActive();
    });

    Viewer.init();
    StatDetail.init();
    RunsView.init();
    Explain.init();
    Research.init();
    Home.init();
    this.views = { home: Home, explain: Explain, runs: RunsView, viewer: Viewer, research: Research };

    // Both layouts are two panes plus a drag handle; the handle sets the width
    // of the second column and the choice is remembered per layout.
    this.makeResizable('viewerLayout', 'viewerResizer', 'gol.width.viewer', 200, 620, 268);

    // Open on whichever view is the default, through the same path a click
    // takes. Marking it in the markup instead would set the class and skip
    // everything else showView does, which is how the wordmark came to be the
    // selected tab without looking like it.
    this.showView(this.view);
    requestAnimationFrame(time => this.frame(time));
  },

  /**
   * One animation loop for the page, handing each frame to the view on
   * screen, with the time since the last one.
   *
   * The front page, the Viewer and the Explanation each ran a loop of their
   * own, all the time, and each asked on every frame whether it was showing.
   * Only the front page caught an error, so one throw in the Viewer or the
   * Explanation stopped its loop for good.
   */
  frame(time) {
    // Never negative, and never a leap: a timestamp can go backwards when a
    // tab is restored, and a long gap would otherwise turn the camera and
    // advance playback by all of it at once.
    const dt = Math.max(0, Math.min(0.1, (time - (this._lastTime ?? time)) / 1000)) || 0;
    this._lastTime = time;
    try {
      this.views[this.view].tick?.(dt, time);
    } catch (err) {
      if (!this._complained) {
        this._complained = true;
        console.warn(`${this.view}:`, err);
      }
    }
    requestAnimationFrame(t => this.frame(t));
  },

  /**
   * Turn a divider into a drag handle that resizes the right-hand column.
   *
   * The width is applied to the grid template rather than the panel itself, so
   * the canvas column reflows and its ResizeObserver picks the change up.
   */
  makeResizable(layoutId, resizerId, storageKey, min, max, fallback) {
    const layout = document.getElementById(layoutId);
    const resizer = document.getElementById(resizerId);
    if (!layout || !resizer) return;

    const clamp = w => Math.max(min, Math.min(max, w));
    const apply = w => { layout.style.gridTemplateColumns = `1fr 6px ${clamp(w)}px`; };

    const stored = Number(localStorage.getItem(storageKey));
    apply(Number.isFinite(stored) && stored > 0 ? stored : fallback);

    let dragging = false;

    resizer.addEventListener('mousedown', e => {
      dragging = true;
      document.body.classList.add('resizing');
      e.preventDefault();
    });

    window.addEventListener('mousemove', e => {
      if (!dragging) return;
      // Measured from the right edge, which is where the panel actually ends.
      apply(layout.getBoundingClientRect().right - e.clientX);
    });

    window.addEventListener('mouseup', () => {
      if (!dragging) return;
      dragging = false;
      document.body.classList.remove('resizing');

      const width = parseInt(layout.style.gridTemplateColumns.split(' ').pop(), 10);
      if (Number.isFinite(width)) localStorage.setItem(storageKey, String(width));
    });

    resizer.addEventListener('dblclick', () => {
      apply(fallback);
      localStorage.setItem(storageKey, String(fallback));
    });
  },

  /**
   * Back from a hidden window: pick up whatever was cut short.
   *
   * Hiding cancels everything, which is right while nobody is looking and
   * wrong the moment somebody is again. Each view knows whether its own last
   * load finished, so this asks rather than reloading what is already drawn.
   */
  reloadActive() {
    this.views[this.view].resume?.();
  },

  showView(name) {
    // Leaving a tab stops what it was loading. Only on an actual change: a
    // second click on the tab you are already on is not a request to throw
    // away the load it is in the middle of.
    if (name !== this.view) Jobs.cancelAll();
    this.view = name;

    for (const tab of document.querySelectorAll('.tab, .brand')) {
      tab.classList.toggle('active', tab.dataset.view === name);
    }
    for (const view of document.querySelectorAll('.view')) {
      view.classList.toggle('active', view.id === `view-${name}`);
    }

    // Every view is told, both ways round: what a view does on the way in and
    // on the way out is its own business.
    for (const [key, view] of Object.entries(this.views)) view.setActive(key === name);
  }
};

document.addEventListener('DOMContentLoaded', () => App.init());
