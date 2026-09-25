/* Tab switching, panel resizing, and start-up. */
const App = {
  view: 'home',

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

    // Both layouts are two panes plus a drag handle; the handle sets the width
    // of the second column and the choice is remembered per layout.
    this.makeResizable('viewerLayout', 'viewerResizer', 'gol.width.viewer', 200, 620, 268);

    // Open on whichever view is the default, through the same path a click
    // takes. Marking it in the markup instead would set the class and skip
    // everything else showView does, which is how the wordmark came to be the
    // selected tab without looking like it.
    this.showView(this.view);
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
    if (this.view === 'research') Research.resume();
    if (this.view === 'viewer') StatDetail.resume();
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

    // The backdrop runs only while it is being looked at, and its canvas has
    // no size until the view is shown, so it is told both ways round.
    Home.setActive(name === 'home');

    // Choosing a backend means, on a static host, downloading a Python
    // runtime. That waits until somebody actually wants to run something.
    if (name === 'runs') RunsView.activate();
    Research.setActive(name === 'research');
    Explain.setActive(name === 'explain');
    Viewer.setActive(name === 'viewer');

    // The canvas has no size while hidden, so it must be measured on reveal.
    // Done synchronously as well as on the next frame: the element already has
    // its box by now, and waiting on rAF alone can leave a blank canvas.
    if (name === 'viewer') {
      StatDetail.resume();
      Viewer.resize();
      if (Viewer.frame) Viewer.updateCharts();
      requestAnimationFrame(() => Viewer.resize());
    }
  },

  isViewerActive() {
    return this.view === 'viewer';
  }
};

document.addEventListener('DOMContentLoaded', () => App.init());
