/* Tab switching and start-up. */
const App = {
  view: 'runs',

  init() {
    for (const tab of document.querySelectorAll('.tab')) {
      tab.addEventListener('click', () => this.showView(tab.dataset.view));
    }

    Viewer.init();
    RunsView.init();
  },

  showView(name) {
    this.view = name;

    for (const tab of document.querySelectorAll('.tab')) {
      tab.classList.toggle('active', tab.dataset.view === name);
    }
    for (const view of document.querySelectorAll('.view')) {
      view.classList.toggle('active', view.id === `view-${name}`);
    }

    // The canvas has no size while hidden, so it must be measured on reveal.
    // Done synchronously as well as on the next frame: the element already has
    // its box by now, and waiting on rAF alone can leave a blank canvas.
    if (name === 'viewer') {
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
